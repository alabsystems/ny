// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! ROUTER_PROGRAM §11.2 tests 1–17 (`ny-core` unit).

use std::sync::mpsc;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use proptest::prelude::*;

use super::bus::{self, ProgressSample};
use super::checkpoint::{self, Amendment, AmendmentLog, CheckpointId};
use super::decl::{
    AmendmentSet, CapId, LaneRole, OpKind, OpSet, ProgressKind, SafePoints, SpecArchetype, WorkUnit,
};
use super::facts::{DeviceQual, FactsStatus, HostClass, HostEnvelope, InstanceFacts};
use super::guard;
use super::hash;
use super::plan::{self, AdmissionState, BudgetView, CapClass, CapInput, LedgerState, PolicyInput};
use super::project::{self, FarClass};
use super::{
    effective, effective_optimizer, EffectiveMode, OptimizerMode, RouterMode, RuleId,
    ACTING_OPTIMIZER_RULES, ACTING_RULES, PRIORS_VERSION, ROUTER_SCHEMA,
};

/// The bus and the fault guard are process globals: serialize the tests that
/// touch them.
static GLOBALS: Mutex<()> = Mutex::new(());

fn globals() -> std::sync::MutexGuard<'static, ()> {
    GLOBALS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// SplitMix64: a dependency-free deterministic generator for the random-input
/// tests (the same seed gives the same inputs on every host).
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: u64) -> u64 {
        if n == 0 {
            0
        } else {
            self.next() % n
        }
    }

    fn coin(&mut self) -> bool {
        self.next() & 1 == 1
    }

    fn opt(&mut self, n: u64) -> Option<u64> {
        self.coin().then(|| self.below(n))
    }
}

const PROGRESS: [ProgressKind; 9] = [
    ProgressKind::FrontierBinary,
    ProgressKind::FrontierKary,
    ProgressKind::Bfs,
    ProgressKind::Volume,
    ProgressKind::Margin,
    ProgressKind::Screen,
    ProgressKind::Dual,
    ProgressKind::Yield,
    ProgressKind::Opaque,
];

fn cap(
    id: &str,
    class: CapClass,
    ledger: LedgerState,
    progress: ProgressKind,
    rung: Option<u64>,
    admission: AdmissionState,
) -> CapInput {
    CapInput {
        id: id.to_owned(),
        class,
        ledger,
        progress,
        unit: WorkUnit::Expansion,
        first_rung_ms: rung,
        safe_points: SafePoints::BetweenUnits,
        accepts: AmendmentSet {
            deadline: true,
            hand_on_source: true,
            hand_on_sink: true,
            defer: false,
        },
        admission,
        route: None,
    }
}

fn host() -> HostEnvelope {
    HostEnvelope {
        class: HostClass::Gb10,
        threads: 20,
        mem_bytes: 121 << 30,
        reserve_bytes: 16 << 30,
        device: DeviceQual::Search,
        declared: false,
    }
}

fn random_input(rng: &mut Rng) -> PolicyInput {
    let mut facts = if rng.coin() {
        InstanceFacts::known_empty()
    } else {
        InstanceFacts::unknown("scan>256MiB")
    };
    facts.ops = OpSet(rng.next());
    facts.ops_complete = rng.coin();
    facts.params = rng.opt(10_000_000);
    facts.conv_layers = rng.opt(40);
    facts.model_bytes = rng.opt(1 << 32);
    facts.declared_networks = rng.opt(3);
    facts.spec = rng.coin().then_some(SpecArchetype::Robustness);
    let n_caps = rng.below(12);
    let capabilities = (0..n_caps)
        .map(|i| {
            let class = match rng.below(5) {
                0 => CapClass::Lane(LaneRole::Attack),
                1 => CapClass::Lane(LaneRole::Screen),
                2 | 3 => CapClass::Lane(LaneRole::Prover {
                    terminal_ok: rng.coin(),
                }),
                _ => CapClass::Tier {
                    parent: "margin-row".to_owned(),
                },
            };
            let ledger = match rng.below(4) {
                0 => LedgerState::Open("S7".to_owned()),
                1 => LedgerState::Shadow,
                2 => LedgerState::SealedAsShipped,
                _ => LedgerState::Sealed,
            };
            let admission = match rng.below(3) {
                0 => AdmissionState::Admit,
                1 => AdmissionState::Decline("predicate".to_owned()),
                _ => AdmissionState::Unknown("inline-admission".to_owned()),
            };
            let progress = PROGRESS[rng.below(PROGRESS.len() as u64) as usize];
            let rung = rng.opt(200_000);
            cap(&format!("cap{i}"), class, ledger, progress, rung, admission)
        })
        .collect();
    let mut pins: Vec<String> = (0..rng.below(4)).map(|i| format!("key.{i}")).collect();
    pins.sort();
    PolicyInput {
        schema: ROUTER_SCHEMA.to_owned(),
        facts,
        host: HostEnvelope {
            threads: rng.below(128) as u32,
            ..host()
        },
        budget: BudgetView {
            official_ms: rng.below(2_000_000),
            internal_ms: rng.below(2_000_000),
            join_margin_ms: rng.below(5_000),
        },
        capabilities,
        pins,
        priors_version: PRIORS_VERSION.to_owned(),
        registry_version: "0123abcd".to_owned(),
        priors: None,
    }
}

/// A row with one FAR-able priced prover `far` (first in order) and a sealed
/// terminal recipient `sink`; window 93 000 ms, c1 23 250, c2 46 500.
fn hand_on_input() -> PolicyInput {
    PolicyInput {
        schema: ROUTER_SCHEMA.to_owned(),
        facts: InstanceFacts::known_empty(),
        host: host(),
        budget: BudgetView {
            official_ms: 100_000,
            internal_ms: 95_000,
            join_margin_ms: 2_000,
        },
        capabilities: vec![
            cap(
                "far",
                CapClass::Lane(LaneRole::Prover { terminal_ok: false }),
                LedgerState::Shadow,
                ProgressKind::FrontierBinary,
                Some(80_000),
                AdmissionState::Admit,
            ),
            cap(
                "sink",
                CapClass::Lane(LaneRole::Prover { terminal_ok: true }),
                LedgerState::Sealed,
                ProgressKind::Opaque,
                None,
                AdmissionState::Admit,
            ),
        ],
        pins: Vec::new(),
        priors_version: PRIORS_VERSION.to_owned(),
        registry_version: "0123abcd".to_owned(),
        priors: None,
    }
}

/// A collapsing frontier for `cap`: no closes while the open set grows.
fn collapsing(cap: &'static str, until_secs: f64) -> Vec<ProgressSample> {
    let steps = 6u64;
    (0..=steps)
        .map(|i| {
            let exp = i * 20;
            ProgressSample::frontier(CapId(cap), exp, 2 * exp + 1, 0, exp + 1)
                .at_secs(until_secs * i as f64 / steps as f64)
        })
        .collect()
}

struct Record {
    source: String,
    verdict: String,
    samples: Vec<ProgressSample>,
}

const FIXTURE_CAP: &str = "margin-row";

fn fixtures() -> Vec<Record> {
    let mut out: Vec<Record> = Vec::new();
    for line in include_str!("fixtures/far_margin_row.txt").lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let fields: Vec<&str> = line.split_whitespace().collect();
        match fields.as_slice() {
            ["record", source, verdict] => {
                out.push(Record {
                    source: (*source).to_owned(),
                    verdict: (*verdict).to_owned(),
                    samples: vec![
                        ProgressSample::frontier(CapId(FIXTURE_CAP), 0, 1, 0, 1).at_secs(0.0)
                    ],
                })
            }
            ["obs", exp, open, created, closed] => {
                let record = out.last_mut().expect("an obs row follows a record header");
                let prev = *record.samples.last().expect("root sample");
                let parse = |s: &str| s.parse::<u64>().expect("fixture integer");
                let sample = ProgressSample::frontier(
                    CapId(FIXTURE_CAP),
                    parse(exp),
                    prev.created + parse(created),
                    prev.closes + parse(closed),
                    parse(open),
                )
                .at_secs(record.samples.len() as f64);
                record.samples.push(sample);
            }
            other => panic!("malformed fixture line {other:?}"),
        }
    }
    assert_eq!(
        out.iter().filter(|r| r.verdict == "unsat").count(),
        8,
        "the fixture must hold the judge's 8 unsat records"
    );
    out
}

/// Every prefix classification of one record's frontier.
fn classifications(record: &Record) -> Vec<FarClass> {
    (0..record.samples.len())
        .map(|k| {
            let trailing = project::trailing(
                &record.samples,
                FIXTURE_CAP,
                k as f64,
                project::FRONTIER_WINDOW_UNITS,
            );
            project::classify(ProgressKind::FrontierBinary, &trailing)
        })
        .collect()
}

#[test]
fn router_mode_on_with_no_acting_rules_is_effectively_shadow() {
    // #router-r2: the binary now carries acting set 1; the mapping itself is
    // unchanged — `on` with an EMPTY acting set is still `shadow`.
    // #router-r4: acting set 3 (CR-L, CR-F) joins it: the router's acting
    // set is R2's seven followed by R4's two, nothing else.
    // #router-r5: acting set 4 (CR-RS, CR-G) follows them.
    let mut expected: Vec<RuleId> = super::acting::R2_RULES.to_vec();
    expected.extend(super::r4::R4_RULES);
    expected.extend(super::route::R5_RULES);
    assert_eq!(
        ACTING_RULES,
        expected.as_slice(),
        "acting sets 1 + 3 + 4 are exactly CR-1, CR-A, CR-C, CR-T, CR-H, CR-R, CR-K, CR-L, \
         CR-F, CR-RS, CR-G"
    );
    // #router-r3: the optimizer now carries acting set 2 (the O-STOP rules).
    // #router-r6: the four O-TRIAL rules follow them.
    let mut optimizer_set: Vec<RuleId> = super::ostop::R3_RULES.to_vec();
    optimizer_set.extend(super::otrial::R6_RULES);
    assert_eq!(
        ACTING_OPTIMIZER_RULES,
        optimizer_set.as_slice(),
        "the optimizer's acting set is exactly rule.governor, rule.ia_yield, \
         rule.attack_defer, rule.root_admission, rule.trial_width, rule.trial_lr_backoff, \
         rule.trial_split_restart, rule.trial_attack_halving"
    );
    assert_eq!(
        effective_optimizer(OptimizerMode::On, &[]),
        EffectiveMode::Shadow
    );
    assert_eq!(effective(RouterMode::On, &[]), EffectiveMode::Shadow);
    assert_eq!(
        effective(RouterMode::On, ACTING_RULES),
        EffectiveMode::Acting,
        "R2: `on` acts through the seven rules"
    );
    assert_eq!(
        effective(RouterMode::Shadow, ACTING_RULES),
        EffectiveMode::Shadow
    );
    assert_eq!(effective(RouterMode::Off, ACTING_RULES), EffectiveMode::Off);
    assert_eq!(
        effective_optimizer(OptimizerMode::On, ACTING_OPTIMIZER_RULES),
        EffectiveMode::Acting,
        "R3: optimizer `on` acts through the four stop rules"
    );
    assert_eq!(
        effective_optimizer(OptimizerMode::Shadow, ACTING_OPTIMIZER_RULES),
        EffectiveMode::Shadow
    );
    assert_eq!(
        effective_optimizer(OptimizerMode::Off, ACTING_OPTIMIZER_RULES),
        EffectiveMode::Off
    );
    // Positive control: the mapping does act once a rule is in the set.
    assert_eq!(
        effective(RouterMode::On, &[RuleId("CR-C")]),
        EffectiveMode::Acting
    );
    for mode in [RouterMode::Off, RouterMode::Shadow, RouterMode::On] {
        assert_eq!(RouterMode::from_token(mode.token()), Some(mode));
    }
    assert_eq!(RouterMode::TOKENS, &["off", "shadow", "on"]);
    assert_eq!(OptimizerMode::TOKENS, RouterMode::TOKENS);
    assert_eq!(RouterMode::default(), RouterMode::Off);
    assert_eq!(OptimizerMode::default(), OptimizerMode::Off);
}

#[test]
fn router_plan_is_pure_and_deterministic() {
    let mut rng = Rng(0x5eed_0001);
    for _ in 0..1_000 {
        let input = random_input(&mut rng);
        let first = plan::plan(&input);
        let again = plan::plan(&input);
        let copied_input = input.clone();
        let cloned = plan::plan(&copied_input);
        assert_eq!(first, again);
        assert_eq!(first, cloned);
        assert_eq!(hash::static_hash(&input), hash::static_hash(&copied_input));
        assert_eq!(first.static_hash, hash::hex(&hash::static_hash(&input)));
        assert!(first.emitted.is_empty(), "R1 emits no value");
        assert!(first.checkpoints.len() <= 3);
    }
}

#[test]
fn router_plan_hash_covers_static_inputs_only() {
    let input = hand_on_input();
    let base = plan::plan(&input);
    let quiet = collapsing("far", 1.0);
    let busy: Vec<ProgressSample> = (0..40u64)
        .map(|i| {
            ProgressSample::frontier(CapId("far"), i * 7, 2 * i * 7 + 1, i * 9, 3)
                .at_secs(i as f64 * 0.1)
        })
        .collect();
    // Samples, rates and amendments never feed the hash.
    for samples in [&quiet, &busy] {
        for at in CheckpointId::ALL {
            let _ = checkpoint::checkpoint(&base, samples, at);
            let _ = checkpoint::observe(&base, samples, at);
        }
        assert_eq!(plan::plan(&input).static_hash, base.static_hash);
    }
    // Every static input moves it.
    let mut facts = input.clone();
    facts.facts.params = Some(1234);
    let mut pins = input.clone();
    pins.pins.push("margin_row.alpha_iters".to_owned());
    let mut budget = input.clone();
    budget.budget.official_ms += 1;
    let mut host_threads = input.clone();
    host_threads.host.threads += 1;
    let mut registry = input.clone();
    registry.registry_version = "fedcba98".to_owned();
    let mut admission = input.clone();
    admission.capabilities[1].admission = AdmissionState::Unknown("inline-admission".to_owned());
    // #admission-precision: the routing shape is a static input too.
    let mut shape = input;
    shape.facts.spec_shape = Some(super::spec::SpecShape {
        atoms: 1,
        ..super::spec::SpecShape::default()
    });
    for (what, perturbed) in [
        ("facts", facts),
        ("spec_shape", shape),
        ("pins", pins),
        ("budget", budget),
        ("host", host_threads),
        ("registry", registry),
        ("admission", admission),
    ] {
        assert_ne!(
            plan::plan(&perturbed).static_hash,
            base.static_hash,
            "perturbing {what} must change the static hash"
        );
    }
}

#[test]
fn router_policy_input_has_no_name_or_path_field() {
    const FORBIDDEN: [&str; 9] = [
        "name",
        "path",
        "file",
        "category",
        "benchmark",
        "instance",
        "onnx",
        "vnnlib",
        "profile",
    ];
    fn keys(value: &serde_json::Value, out: &mut Vec<String>) {
        match value {
            serde_json::Value::Object(map) => {
                for (key, child) in map {
                    out.push(key.clone());
                    keys(child, out);
                }
            }
            serde_json::Value::Array(items) => items.iter().for_each(|v| keys(v, out)),
            _ => {}
        }
    }
    // A fully populated input: every optional field present, every enum shape.
    let mut input = hand_on_input();
    input.facts = InstanceFacts {
        status: FactsStatus::Unknown("x".to_owned()),
        ops: OpSet::EMPTY.with(OpKind::Conv).with(OpKind::Relu),
        ops_complete: true,
        params: Some(1),
        nonlinear: Some(2),
        relu: Some(3),
        conv_layers: Some(4),
        max_conv_channels: Some(5),
        p_eff: Some(6),
        input_dims: Some(7),
        spec: Some(SpecArchetype::Disjunctive),
        classes: Some(8),
        clauses: Some(9),
        model_bytes: Some(10),
        declared_networks: Some(2),
        // #admission-precision: every routing-shape field present.
        spec_shape: Some(super::spec::SpecShape {
            atoms: 9,
            clauses: 9,
            max_clause_atoms: 1,
            boxed_clauses: 1,
            disjunction: true,
            relational_atoms: true,
            argmax_margin: true,
            multi_clause_disjunction: true,
            boxed_clause_disjunction: true,
            same_lhs_reducible: true,
            inputs: 3072,
            perturbed_inputs: 3072,
            outputs: 10,
        }),
    };
    input.pins = vec!["k".to_owned()];
    input.capabilities.push(cap(
        "t",
        CapClass::Tier {
            parent: "p".to_owned(),
        },
        LedgerState::Open("S7".to_owned()),
        ProgressKind::Bfs,
        Some(1),
        AdmissionState::Decline("d".to_owned()),
    ));
    input.capabilities.push(cap(
        "v",
        CapClass::Variant {
            slot: "s".to_owned(),
        },
        LedgerState::SealedAsShipped,
        ProgressKind::Yield,
        None,
        AdmissionState::Unknown("u".to_owned()),
    ));
    input.capabilities.push(cap(
        "ph",
        CapClass::Phase {
            lane: "l".to_owned(),
        },
        LedgerState::Shadow,
        ProgressKind::Volume,
        None,
        AdmissionState::Admit,
    ));
    input.capabilities.push(cap(
        "r",
        CapClass::Rule,
        LedgerState::Shadow,
        ProgressKind::Opaque,
        None,
        AdmissionState::Admit,
    ));
    // #router-r7: the offline prior's view, every field populated.
    input.priors = Some(super::priors::PriorView {
        version: "cost-v1+00000000".to_owned(),
        host: "gb10".to_owned(),
        cell: Some("conv=1;nonpwl=0".to_owned()),
        predictions: vec![super::priors::PhasePrediction {
            phase: "verifier.setup".to_owned(),
            cell: "*".to_owned(),
            median_ms: 1,
            upper_ms: 2,
        }],
        priced: vec![super::priors::PricedCap {
            cap: "c".to_owned(),
            ms: 3,
        }],
        unpriced: vec![super::priors::UnpricedCap {
            cap: "d".to_owned(),
            reason: "host:m5".to_owned(),
        }],
    });
    // #admission-precision: a routable capability with a site.
    input.capabilities.push(CapInput {
        route: Some(plan::RouteSpec {
            sr: super::decl::SrLevel::Decl,
            parent: Some("p".to_owned()),
            admit_when: super::decl::AdmitWhen {
                site: Some(super::decl::AdmitSite::DagGraph),
                ..super::decl::AdmitWhen::ANY
            },
            envelope_ops: OpSet::EMPTY,
            envelope_specs: super::decl::SpecSet::EMPTY,
            risk: super::decl::KnobRisk::High,
            settings: vec![("k".to_owned(), "v".to_owned())],
            slot: Some("s".to_owned()),
        }),
        ..cap(
            "rt",
            CapClass::Phase {
                lane: "l".to_owned(),
            },
            LedgerState::Open("S7".to_owned()),
            ProgressKind::Opaque,
            None,
            AdmissionState::Admit,
        )
    });
    let json = serde_json::to_value(&input).expect("the input serializes");
    let mut all = Vec::new();
    keys(&json, &mut all);
    assert!(
        all.len() > 40,
        "the key walk must see the whole input: {all:?}"
    );
    for key in &all {
        let lower = key.to_ascii_lowercase();
        for needle in FORBIDDEN {
            assert!(
                !lower.contains(needle),
                "PolicyInput carries a `{key}` field: the policy must be name- and path-blind by type"
            );
        }
    }
    // And the round trip is exact (replay depends on it).
    let back: PolicyInput = serde_json::from_value(json).expect("the input deserializes");
    assert_eq!(back, input);
}

#[test]
fn router_contracts_sum_within_window_minus_reserves() {
    let mut rng = Rng(0x5eed_0005);
    for _ in 0..2_000 {
        let input = random_input(&mut rng);
        let p = plan::plan(&input);
        let official = input.budget.official_ms;
        let expected_window = official
            .saturating_sub(plan::publication_floor_ms(official))
            .saturating_sub(input.budget.join_margin_ms);
        assert_eq!(p.window_ms, expected_window);
        let priced: u64 = p.contracts.iter().filter_map(|c| c.ms).sum();
        assert!(
            priced <= p.window_ms,
            "contracts {priced}ms exceed the window {}ms (= official − r_pub − join)",
            p.window_ms
        );
        assert!(p.contracts.iter().filter(|c| c.terminal).count() <= 1);
        if let Some(terminal) = &p.terminal {
            assert_eq!(
                p.order.last(),
                Some(terminal),
                "the terminal lane runs last"
            );
        }
    }
}

#[test]
fn router_never_plans_below_publication_floor() {
    let mut rng = Rng(0x5eed_0006);
    let mut budgets: Vec<u64> = vec![
        0,
        1,
        4_999,
        5_000,
        99_999,
        100_000,
        20 * 5_000 + 1,
        u64::MAX,
    ];
    budgets.extend((0..500).map(|_| rng.next() >> rng.below(64)));
    for official in budgets {
        let mut input = hand_on_input();
        input.budget.official_ms = official;
        let p = plan::plan(&input);
        let floor = (official / 20).max(5_000);
        assert!(
            p.reserves.r_pub_ms >= floor,
            "official={official}: r_pub {} below max(5 s, B/20) = {floor}",
            p.reserves.r_pub_ms
        );
        assert!(p.window_ms <= official.saturating_sub(floor));
    }
}

#[test]
fn router_checkpoint_c1_never_amends() {
    let input = hand_on_input();
    let p = plan::plan(&input);
    let samples = collapsing("far", 10.0);
    // Positive control: the same samples DO amend at c2, so c1's silence is
    // not vacuous.
    let c2 = checkpoint::checkpoint(&p, &samples, CheckpointId::C2);
    assert!(
        matches!(c2.as_slice(), [Amendment::HandOn { from, to, ms }] if from == "far" && to == "sink" && *ms == 80_000 - 46_500),
        "control: a FAR lane with a sealed recipient hands on at c2: {c2:?}"
    );
    assert_eq!(
        checkpoint::checkpoint(&p, &samples, CheckpointId::C1).len(),
        0
    );
    assert!(
        checkpoint::observe(&p, &samples, CheckpointId::C1)
            .iter()
            .any(|o| o.class == FarClass::Far),
        "c1 still OBSERVES (and logs) the classification"
    );
    let mut rng = Rng(0x5eed_0007);
    for _ in 0..500 {
        let input = random_input(&mut rng);
        let p = plan::plan(&input);
        let samples: Vec<ProgressSample> = input
            .capabilities
            .iter()
            .flat_map(|c| {
                let id: &'static str = match c.id.as_str() {
                    "cap0" => "cap0",
                    "cap1" => "cap1",
                    "cap2" => "cap2",
                    _ => "cap3",
                };
                collapsing(id, 5.0)
            })
            .collect();
        assert_eq!(
            checkpoint::checkpoint(&p, &samples, CheckpointId::C1).len(),
            0
        );
    }
}

#[test]
fn router_far_rule_flags_no_unsat_fixture() {
    let records = fixtures();
    let mut windows_seen = 0usize;
    for record in records.iter().filter(|r| r.verdict == "unsat") {
        for (k, class) in classifications(record).into_iter().enumerate() {
            assert_ne!(
                class,
                FarClass::Far,
                "{} (unsat) classified FAR at observation {k}",
                record.source
            );
            if class != FarClass::Unknown {
                windows_seen += 1;
            }
        }
    }
    assert!(
        windows_seen >= 8,
        "the unsat fixtures must actually be classified ({windows_seen})"
    );
    // FD t2712 (w10b/v2/FDB_t2712_fdb_r1.raw, class 183): the one closed class
    // of the row logged nodes=248 (root excluded) certified_prunes=125
    // depth=14. The level ledger is not printed, so only the class total is on
    // record: the frontier identity reads it closed, and a class-level sample
    // is never FAR.
    assert_eq!(project::bfs_open(248 + 1, 125), Some(0));
    let class_sample = [ProgressSample::bfs(CapId("frozen-dual-bab"), 14, 249, 125).at_secs(0.056)];
    assert_ne!(
        project::classify(ProgressKind::Bfs, &class_sample),
        FarClass::Far
    );
}

#[test]
fn router_far_rule_flags_collapsed_timeouts() {
    let records = fixtures();
    let fired = |source: &str| {
        let record = records
            .iter()
            .find(|r| r.source.ends_with(source))
            .unwrap_or_else(|| panic!("fixture {source} missing"));
        assert_eq!(record.verdict, "timeout");
        classifications(record)
            .iter()
            .position(|c| *c == FarClass::Far)
    };
    let c8330 = fired("MRG_c8330_ctl_r1").expect("c8330 r1 collapses: FAR must fire");
    assert!(
        c8330 >= 8,
        "c8330 fires only on its late windows (fired at {c8330})"
    );
    assert!(
        fired("MRG_c8330_ctl_r2").is_some(),
        "c8330 r2 collapses too"
    );
    assert!(
        fired("CCT_1440_w_r1").is_some(),
        "cct c1440 never closes: FAR must fire"
    );
    // MEASURED correction to the brief: c3494's trailing windows read
    // 0.105 / 0.333 / 0.100 closes/expansion, never two consecutive at or
    // below T_FAR = 0.10, so the pre-registered rule does NOT flag it.
    assert_eq!(fired("FDB_c3494_ctl_r1"), None);
}

#[test]
fn router_amendment_has_no_promotion_variant() {
    let source = include_str!("checkpoint.rs");
    let code: String = source
        .lines()
        .filter(|l| !l.trim_start().starts_with("//"))
        .collect::<Vec<_>>()
        .join("\n");
    assert!(code.len() > 500, "checkpoint.rs is suspiciously short");
    for needle in [
        format!("{}{}", "Pro", "mote"),
        format!("{}{}", "Pro", "motion"),
        format!("{}{}", "Ne", "ar"),
    ] {
        assert!(
            !code.contains(needle.as_str()),
            "checkpoint.rs names `{needle}`: there is no NEAR class and no promotion"
        );
    }
    let start = code
        .find("pub enum Amendment {")
        .expect("the Amendment enum");
    let body = &code[start..];
    let body = &body[..body.find("\n}").expect("closed enum")];
    let variants: Vec<&str> = body
        .lines()
        .skip(1)
        .filter(|l| l.starts_with("    ") && !l.starts_with("     "))
        .map(|l| l.trim().trim_end_matches(['{', ',', ' ']))
        .filter(|l| !l.is_empty() && !l.starts_with('#') && !l.starts_with('}'))
        .collect();
    assert_eq!(
        variants,
        ["HandOn", "Defer", "ArmPublishGrace", "DeriveCap"]
    );
}

#[test]
fn router_bfs_frontier_identity() {
    // t4863 class 175 (op_fd_tier_census.out): N=354,482 P=146,232.
    assert_eq!(project::bfs_open(354_482, 146_232), Some(62_019));
    let growth = project::bfs_growth(354_482, 146_232).expect("growth");
    assert!(growth > 1.0, "t4863 c175 was still growing (g = {growth})");
    assert!((growth - 1.175).abs() < 1e-3);
    // Inconsistent counters are unknown, never a wrapped integer.
    assert_eq!(project::bfs_open(1, 2), None);
    assert_eq!(project::bfs_open(u64::MAX, 0), None);
    assert_eq!(project::bfs_growth(0, 0), None);

    proptest!(ProptestConfig::with_cases(256), |(decisions in proptest::collection::vec(any::<bool>(), 1..400))| {
        // Grow a random full binary tree: pop an open node, prune it or split
        // it (decision), until the open set empties or the decisions run out.
        let (mut open, mut evaluated, mut pruned) = (1u64, 0u64, 0u64);
        for split in decisions {
            if open == 0 {
                break;
            }
            open -= 1;
            evaluated += 1;
            if split {
                open += 2;
            } else {
                pruned += 1;
            }
            prop_assert_eq!(project::bfs_open(evaluated, pruned), Some(open));
        }
        // Close the tree: prune everything left.
        while open > 0 {
            open -= 1;
            evaluated += 1;
            pruned += 1;
        }
        prop_assert_eq!(project::bfs_open(evaluated, pruned), Some(0));
    });
}

#[test]
fn router_margin_projection_uses_rate_not_level() {
    let trace = |start: f64, slope: f64| -> Vec<(f64, f64)> {
        (0..=10)
            .map(|i| {
                let t = f64::from(i);
                (t, start + slope * t)
            })
            .collect()
    };
    let flat = trace(-0.033, 0.0);
    let climbing = trace(-0.083, 0.006);
    assert_eq!(
        project::far_margin(&flat, 10.0, project::MARGIN_NOISE_EPS),
        FarClass::Far
    );
    assert_eq!(
        project::far_margin(&climbing, 10.0, project::MARGIN_NOISE_EPS),
        FarClass::NotFar
    );
    // The flat trace sits CLOSER to zero: the level alone would rank it better.
    assert!(flat[10].1 > climbing[0].1);
    // A trace spanning less than half the slice is not evidence.
    assert_eq!(
        project::far_margin(&flat[..3], 10.0, project::MARGIN_NOISE_EPS),
        FarClass::Unknown
    );
    // Through the sample path too.
    let samples: Vec<ProgressSample> = [(0.0, -0.033), (5.0, -0.033), (10.0, -0.033)]
        .into_iter()
        .enumerate()
        .map(|(i, (t, m))| ProgressSample::scalar(CapId("attack"), i as u64, m).at_secs(t))
        .collect();
    assert_eq!(
        project::classify(ProgressKind::Margin, &samples),
        FarClass::Far
    );
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]
    #[test]
    fn router_arithmetic_never_panics_on_extremes(
        official in prop_oneof![Just(0u64), Just(u64::MAX), any::<u64>()],
        join in prop_oneof![Just(0u64), Just(u64::MAX), any::<u64>()],
        rung in prop_oneof![Just(0u64), Just(u64::MAX), any::<u64>()],
        counter in prop_oneof![Just(0u64), Just(u64::MAX), any::<u64>()],
        real in prop_oneof![
            Just(f64::NAN),
            Just(f64::INFINITY),
            Just(f64::NEG_INFINITY),
            Just(0.0f64),
            any::<f64>()
        ],
    ) {
        let mut input = hand_on_input();
        input.budget.official_ms = official;
        input.budget.join_margin_ms = join;
        input.budget.internal_ms = counter;
        input.capabilities[0].first_rung_ms = Some(rung);
        input.host.mem_bytes = counter;
        input.host.reserve_bytes = u64::MAX;
        let p = plan::plan(&input);
        let samples: Vec<ProgressSample> = (0..5u64)
            .map(|i| ProgressSample {
                cap: CapId("far"),
                unit_index: counter,
                work_done: counter.wrapping_add(i),
                closes: counter,
                open: counter.wrapping_sub(i),
                created: counter,
                level: counter,
                pruned: counter.wrapping_mul(i),
                margin_best_bits: real.to_bits(),
                secs_bits: real.to_bits(),
            })
            .collect();
        for at in CheckpointId::ALL {
            let _ = checkpoint::checkpoint(&p, &samples, at);
            let _ = checkpoint::observe(&p, &samples, at);
            let _ = checkpoint::select_samples(&p, &samples, at);
        }
        for kind in PROGRESS {
            let _ = project::classify(kind, &samples);
        }
        let _ = project::bfs_open(counter, counter);
        let _ = project::bfs_growth(counter, rung);
        let _ = project::far_margin(&[(real, real), (0.0, real)], real, real);
        let _ = project::far_screen(real, real, real);
        let _ = project::margin_slope(&[(real, 1.0), (real, 2.0)]);
        let _ = project::governor_yield_index(&[real, 1.0, real, 0.0], real, 2);
        let _ = checkpoint::contract_end_ms(&p, "far");
        let _ = hash::static_hash(&input);
    }
}

#[test]
fn router_fail_open_contains_injected_panic() {
    let _serial = globals();
    guard::reset();
    bus::reset();
    bus::enable(Instant::now());
    let fault = guard::fail_open("injected", || -> u32 { panic!("router-r1 injected panic") })
        .expect_err("the injected panic is contained");
    assert_eq!(fault.site, "injected");
    assert!(!fault.skipped);
    assert!(fault.message.contains("injected panic"), "{fault}");
    assert_eq!(guard::faults(), 1);
    assert!(guard::disabled(), "the first fault disables the router");
    assert!(!bus::enabled(), "and stops the bus");
    // The session is disabled: a later entry does not run at all.
    let mut ran = false;
    let skipped = guard::fail_open("after", || ran = true).expect_err("disabled");
    assert!(skipped.skipped);
    assert!(!ran, "a disabled router runs nothing");
    assert_eq!(guard::faults(), 1, "a skipped call is not a second fault");
    // The caller continues normally.
    guard::reset();
    assert_eq!(guard::fail_open("clean", || 7).expect("clean call"), 7);
    bus::reset();
}

/// #router-r2 §6.3: a contained router fault fails open to the static plan —
/// it DISARMS the one deadline authority, so every lane reads `None` from its
/// cell and keeps the deadline its own rule derives today, and no publish
/// grace survives.
#[test]
fn router_r2_fault_disarms_the_deadline_authority() {
    use crate::deadline_authority::{self, Frame, LaneSlot, SLOTS};
    let _serial = globals();
    guard::reset();
    let authority = deadline_authority::global();
    let anchor = Instant::now();
    let frame = Frame::new(
        anchor,
        anchor + Duration::from_secs(100),
        [Duration::from_secs(5); SLOTS],
        Duration::from_secs(2),
    );
    authority
        .arm(
            frame,
            [true; SLOTS],
            Some(LaneSlot::Verifier),
            Some(Duration::from_millis(1_500)),
        )
        .expect("arm");
    assert!(
        authority.governs(LaneSlot::AttackUpfront),
        "positive control"
    );
    assert_eq!(
        authority.publish_grace(),
        Some(Duration::from_millis(1_500))
    );
    let fault = guard::fail_open("injected-r2", || -> u32 {
        panic!("router-r2 injected panic")
    })
    .expect_err("the injected panic is contained");
    assert!(!fault.skipped);
    assert!(
        !authority.is_armed(),
        "a contained fault disarms the authority"
    );
    for slot in LaneSlot::ALL {
        assert!(!authority.governs(slot), "{slot:?}");
        assert_eq!(authority.deadline(slot), None, "{slot:?}");
        assert_eq!(
            authority.commit_contract(slot, Duration::from_secs(3), anchor),
            None
        );
    }
    assert_eq!(authority.publish_grace(), None, "no grace survives a fault");
    guard::reset();
    bus::reset();
}

#[test]
fn progress_bus_is_noop_when_disabled() {
    let _serial = globals();
    bus::reset();
    assert!(!bus::enabled());
    let stride = bus::Stride::new();
    for i in 0..1_000u64 {
        bus::emit(ProgressSample::frontier(CapId("x"), i, i, i, i));
        assert!(!stride.due(i, 1), "a dark tap is never due");
    }
    assert_eq!(bus::stats(), bus::BusStats::default());
    assert_eq!(bus::drain().len(), 0);
    // Control: enabled, the same pushes land.
    bus::enable(Instant::now());
    bus::emit(ProgressSample::frontier(CapId("x"), 1, 1, 1, 1));
    assert_eq!(bus::stats().pushed, 1);
    assert_eq!(bus::drain().len(), 1);
    bus::reset();
}

#[test]
fn progress_bus_writers_never_block() {
    const WRITERS: usize = 8;
    const PUSHES: u64 = 200;
    let _serial = globals();
    bus::reset();
    bus::enable(Instant::now());
    let (tx, rx) = mpsc::channel::<usize>();
    let mut handles = Vec::new();
    let all_done = bus::with_ring_locked(|| {
        for w in 0..WRITERS {
            let tx = tx.clone();
            handles.push(std::thread::spawn(move || {
                for i in 0..PUSHES {
                    bus::emit(ProgressSample::frontier(CapId("w"), i, i, 0, 0));
                }
                let _ = tx.send(w);
            }));
        }
        // Every writer must FINISH while the reader still holds the lock: a
        // writer that waited on the lock would never report.
        let deadline = Instant::now() + Duration::from_secs(30);
        let mut done = 0usize;
        while done < WRITERS {
            let left = deadline.saturating_duration_since(Instant::now());
            match rx.recv_timeout(left) {
                Ok(_) => done += 1,
                Err(_) => break,
            }
        }
        done
    });
    for handle in handles {
        handle.join().expect("writer thread");
    }
    assert_eq!(all_done, WRITERS, "a writer blocked on the reader's lock");
    let stats = bus::stats();
    assert_eq!(stats.pushed, 0);
    assert_eq!(
        stats.dropped,
        WRITERS as u64 * PUSHES,
        "every contended push is counted"
    );
    bus::reset();
}

#[test]
fn router_checkpoint_amendments_are_append_only_and_bounded() {
    // Twenty FAR-able priced provers and one sealed terminal recipient; on the
    // sequential schedule nine contracts are still running at c2.
    let mut input = hand_on_input();
    input.budget.official_ms = 1_000_000;
    input.capabilities.clear();
    const IDS: [&str; 20] = [
        "f00", "f01", "f02", "f03", "f04", "f05", "f06", "f07", "f08", "f09", "f10", "f11", "f12",
        "f13", "f14", "f15", "f16", "f17", "f18", "f19",
    ];
    for id in IDS {
        input.capabilities.push(cap(
            id,
            CapClass::Lane(LaneRole::Prover { terminal_ok: false }),
            LedgerState::Shadow,
            ProgressKind::FrontierBinary,
            Some(40_000),
            AdmissionState::Admit,
        ));
    }
    input.capabilities.push(cap(
        "sink",
        CapClass::Lane(LaneRole::Prover { terminal_ok: true }),
        LedgerState::Sealed,
        ProgressKind::Opaque,
        None,
        AdmissionState::Admit,
    ));
    let p = plan::plan(&input);
    let samples: Vec<ProgressSample> = IDS.iter().flat_map(|id| collapsing(id, 3.0)).collect();
    let c2 = checkpoint::checkpoint(&p, &samples, CheckpointId::C2);
    assert_eq!(
        c2.len(),
        checkpoint::MAX_AMENDMENTS_PER_CHECKPOINT,
        "nine lanes qualify; one checkpoint is capped"
    );
    let c3 = checkpoint::checkpoint(&p, &samples, CheckpointId::C3);
    assert!(c3.len() <= checkpoint::MAX_AMENDMENTS_PER_CHECKPOINT);

    let mut log = AmendmentLog::new();
    let c2_appended = log.append(CheckpointId::C2, c2.clone());
    let after_c2: Vec<(CheckpointId, Amendment)> = log.entries().to_vec();
    assert_eq!(c2_appended, c2.len());
    let c3_appended = log.append(CheckpointId::C3, c3.clone());
    assert_eq!(c3_appended, c3.len());
    assert_eq!(
        &log.entries()[..after_c2.len()],
        after_c2.as_slice(),
        "c3 only appends"
    );
    // An earlier checkpoint cannot be appended after a later one.
    let before = log.entries().len();
    assert_eq!(log.append(CheckpointId::C2, c2), 0);
    assert_eq!(log.entries().len(), before);
    // Oversized batches are capped per checkpoint and in total.
    let flood: Vec<Amendment> = (0..100u64)
        .map(|i| Amendment::Defer {
            from: format!("x{i}"),
            ms: i,
        })
        .collect();
    let mut bounded = AmendmentLog::new();
    for at in CheckpointId::ALL {
        bounded.append(at, flood.clone());
    }
    assert!(bounded.entries().len() <= checkpoint::MAX_AMENDMENTS_TOTAL);
    assert_eq!(
        bounded.entries().len() as u64 + bounded.dropped(),
        300,
        "every refused entry is counted"
    );
}

#[test]
fn router_trailing_selection_is_idempotent() {
    // Supports replay: a flight record carries only the selected samples.
    let record = fixtures()
        .into_iter()
        .find(|r| r.source.ends_with("MRG_c2050_ctl_r1"))
        .expect("fixture");
    for k in 0..record.samples.len() {
        let t = k as f64;
        let once = project::trailing(&record.samples, FIXTURE_CAP, t, 16);
        let twice = project::trailing(&once, FIXTURE_CAP, t, 16);
        assert_eq!(once, twice);
        assert_eq!(
            project::classify(ProgressKind::FrontierBinary, &once),
            project::classify(ProgressKind::FrontierBinary, &twice)
        );
    }
}

// ============================================================================
// #router-r2 — acting set 1 (ROUTER_PROGRAM §10 R2), rule by rule.
// ============================================================================

mod r2 {
    use std::time::{Duration, Instant};

    use super::super::acting::{
        self, Commitments, RuleSwitches, R2_RULES, SLOT_CAPS, TODAY_MAX_DEPTH,
    };
    use super::super::derive::{self, DepthGate};
    use super::super::plan::{
        self, AdmissionState, BudgetView, CapClass, LedgerState, PolicyInput,
    };
    use super::super::{EffectiveMode, PRIORS_VERSION, ROUTER_SCHEMA};
    use super::{cap, host, random_input, Rng};
    use crate::deadline_authority::{Authority, Frame, LaneSlot, SLOTS};
    use crate::router::decl::{LaneRole, ProgressKind};
    use crate::router::facts::InstanceFacts;

    const MODES: [EffectiveMode; 3] = [
        EffectiveMode::Off,
        EffectiveMode::Shadow,
        EffectiveMode::Acting,
    ];

    /// The registry-shaped row: every slot's capability, each with an inline
    /// (deferred) admission and a sealed-as-shipped ledger, as on a real row.
    fn registry_row(official_ms: u64, internal_ms: u64, join_ms: u64) -> PolicyInput {
        let lane = |id: &str, role: LaneRole| {
            cap(
                id,
                CapClass::Lane(role),
                LedgerState::SealedAsShipped,
                ProgressKind::Opaque,
                None,
                AdmissionState::Unknown("inline-admission".to_owned()),
            )
        };
        let prover = LaneRole::Prover { terminal_ok: true };
        PolicyInput {
            schema: ROUTER_SCHEMA.to_owned(),
            facts: InstanceFacts::known_empty(),
            host: host(),
            budget: BudgetView {
                official_ms,
                internal_ms,
                join_margin_ms: join_ms,
            },
            capabilities: vec![
                lane("attack.upfront_apgd", LaneRole::Attack),
                lane("attack.bnn_lp", LaneRole::Attack),
                lane("attack.bnn_ste", LaneRole::Attack),
                lane("mip.ay", prover),
                lane("input_split", prover),
                lane("lowdim", prover),
                lane("relu_split.graph", prover),
                lane("margin-row", prover),
            ],
            pins: Vec::new(),
            priors_version: PRIORS_VERSION.to_owned(),
            registry_version: "0123abcd".to_owned(),
            priors: None,
        }
    }

    fn commit_row(
        input: &PolicyInput,
        mode: EffectiveMode,
        switches: &RuleSwitches,
    ) -> Commitments {
        acting::commit(&plan::plan(input), input, mode, switches)
    }

    fn rule<'a>(c: &'a Commitments, id: &str) -> &'a acting::RuleAct {
        c.rules
            .iter()
            .find(|act| act.rule == id)
            .unwrap_or_else(|| panic!("rule {id} missing from {:?}", c.rules))
    }

    fn switches_without(rule: &str) -> RuleSwitches {
        let mut s = RuleSwitches::ALL_ON;
        match rule {
            "CR-A" => s.admission = false,
            "CR-C" => s.contracts = false,
            "CR-1" => s.cap_authority = false,
            "CR-T" => s.terminal = false,
            "CR-H" => s.hand_on = false,
            "CR-R" => s.publish_grace = false,
            "CR-K" => s.derived_caps = false,
            other => panic!("unknown rule {other}"),
        }
        s
    }

    #[test]
    fn router_r2_commitments_are_pure_and_deterministic() {
        let mut rng = Rng(0x5eed_2201);
        for _ in 0..500 {
            let input = random_input(&mut rng);
            let plan = plan::plan(&input);
            for mode in MODES {
                let switches = RuleSwitches {
                    admission: rng.coin(),
                    contracts: rng.coin(),
                    cap_authority: rng.coin(),
                    terminal: rng.coin(),
                    hand_on: rng.coin(),
                    publish_grace: rng.coin(),
                    derived_caps: rng.coin(),
                };
                let a = acting::commit(&plan, &input, mode, &switches);
                let b = acting::commit(&plan, &input, mode, &switches);
                assert_eq!(a, b, "commit is a pure function of its input");
                assert_eq!(a.sha.len(), 64);
                assert_eq!(a.rules.len(), R2_RULES.len(), "every rule decides");
                for (act, id) in a.rules.iter().zip(R2_RULES) {
                    assert_eq!(act.rule, id.as_str(), "rules are reported in R2 order");
                    assert_eq!(act.acts, act.blocked_by.is_none(), "{act:?}");
                }
            }
        }
    }

    #[test]
    fn router_r2_nothing_acts_unless_the_mode_is_acting() {
        let input = registry_row(100_000, 95_000, 2_000);
        for mode in [EffectiveMode::Off, EffectiveMode::Shadow] {
            let c = commit_row(&input, mode, &RuleSwitches::ALL_ON);
            assert!(!c.acting);
            assert!(!c.arm_authority, "{mode:?} never arms the authority");
            assert_eq!(c.governance.len(), 0);
            assert_eq!(c.terminal, None);
            assert!(!c.attack_contracts && !c.release_hand_on);
            assert_eq!(c.grace_ms, None);
            assert_eq!(c.derived, acting::DerivedPlan::default());
            for act in &c.rules {
                assert!(!act.acts, "{mode:?}: {act:?}");
                assert_eq!(
                    act.blocked_by.as_deref(),
                    Some(format!("mode:{}", mode.token()).as_str())
                );
            }
        }
        let on = commit_row(&input, EffectiveMode::Acting, &RuleSwitches::ALL_ON);
        assert!(on.acting && on.arm_authority, "positive control: `on` acts");
        assert!(R2_RULES.iter().all(|id| on.acts(*id)), "{:?}", on.rules);
    }

    #[test]
    fn router_r2_cap_authority_rides_every_cell_rule() {
        let input = registry_row(100_000, 95_000, 2_000);
        let c = commit_row(&input, EffectiveMode::Acting, &switches_without("CR-1"));
        assert!(!c.arm_authority);
        assert_eq!(c.governance.len(), 0);
        for id in ["CR-A", "CR-C", "CR-T", "CR-R"] {
            let act = rule(&c, id);
            assert!(!act.acts, "{id} rides on the authority: {act:?}");
            assert_eq!(act.blocked_by.as_deref(), Some("CR-1"), "{id}");
        }
        // The rule-owned configuration values do not need a cell.
        assert!(rule(&c, "CR-H").acts && c.derived.warmup_hand_on);
        assert!(
            !c.release_hand_on,
            "no terminal, so releases return to the pool"
        );
        assert!(rule(&c, "CR-K").acts && c.derived.clause_passes.is_some());
    }

    #[test]
    fn router_r2_each_rule_key_ablates_its_own_rule() {
        let input = registry_row(100_000, 95_000, 2_000);
        let all = commit_row(&input, EffectiveMode::Acting, &RuleSwitches::ALL_ON);
        for id in R2_RULES {
            let c = commit_row(
                &input,
                EffectiveMode::Acting,
                &switches_without(id.as_str()),
            );
            let own = rule(&c, id.as_str());
            assert!(!own.acts, "{id:?} off must not act");
            assert_eq!(own.blocked_by.as_deref(), Some("key"), "{id:?}");
            assert_ne!(c.sha, all.sha, "{id:?} off must change the commitments");
            // A rule that does not ride on the ablated one is untouched.
            for other in R2_RULES {
                let rides_on_authority =
                    id == plan::CR_1 && ["CR-A", "CR-C", "CR-T", "CR-R"].contains(&other.as_str());
                if other != id && !rides_on_authority {
                    assert_eq!(
                        rule(&c, other.as_str()).acts,
                        rule(&all, other.as_str()).acts,
                        "ablating {id:?} moved {other:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn router_r2_admission_governs_by_the_lanes_own_answer() {
        let base = registry_row(100_000, 95_000, 2_000);
        let governed = |input: &PolicyInput, slot: LaneSlot| {
            commit_row(input, EffectiveMode::Acting, &RuleSwitches::ALL_ON).governed(slot)
        };
        // Inline admissions DEFER to the lane's own site: governed.
        for slot in LaneSlot::ALL {
            assert!(governed(&base, slot), "{slot:?} deferred must be governed");
        }
        let c = commit_row(&base, EffectiveMode::Acting, &RuleSwitches::ALL_ON);
        assert!(
            c.governance
                .iter()
                .all(|g| g.reason.starts_with("deferred:")),
            "{:?}",
            c.governance
        );
        // A declined lane keeps today's deadline.
        let mut declined = base.clone();
        for cap in &mut declined.capabilities {
            if cap.id == "attack.bnn_lp" {
                cap.admission = AdmissionState::Decline("no-sign".to_owned());
            }
        }
        assert!(!governed(&declined, LaneSlot::AttackSignSpace));
        assert!(governed(&declined, LaneSlot::AttackUpfront));
        // An open ledger is never governed.
        let mut open = base.clone();
        for cap in &mut open.capabilities {
            if cap.id == "margin-row" {
                cap.ledger = LedgerState::Open("S7".to_owned());
            }
        }
        assert!(!governed(&open, LaneSlot::MarginRowConcurrent));
        assert!(!governed(&open, LaneSlot::MarginRowInline));
        // The verifier stays governed until EVERY route declined.
        let mut routes = base.clone();
        for cap in &mut routes.capabilities {
            if ["mip.ay", "input_split", "lowdim"].contains(&cap.id.as_str()) {
                cap.admission = AdmissionState::Decline("route".to_owned());
            }
        }
        assert!(governed(&routes, LaneSlot::Verifier));
        for cap in &mut routes.capabilities {
            if cap.id == "relu_split.graph" {
                cap.admission = AdmissionState::Decline("route".to_owned());
            }
        }
        assert!(!governed(&routes, LaneSlot::Verifier));
        // A first rung that does not fit is ungoverned (the plan prices only
        // admitted lanes, so the lane admits and its rung overflows the window).
        let mut unfit = base.clone();
        for cap in &mut unfit.capabilities {
            if cap.id == "attack.bnn_ste" {
                cap.admission = AdmissionState::Admit;
                cap.first_rung_ms = Some(10_000_000);
            }
        }
        assert!(!governed(&unfit, LaneSlot::AttackStePgd));
        // Unregistered: a slot whose capability is absent is ungoverned.
        let mut missing = base;
        missing
            .capabilities
            .retain(|cap| cap.id != "attack.upfront_apgd");
        assert!(!governed(&missing, LaneSlot::AttackUpfront));
        // CR-A off: the authority governs every slot.
        let off = commit_row(&declined, EffectiveMode::Acting, &switches_without("CR-A"));
        assert!(off.governed(LaneSlot::AttackSignSpace));
        assert_eq!(SLOT_CAPS.len(), SLOTS);
    }

    #[test]
    fn router_r2_contracts_and_terminal_follow_governance() {
        let mut input = registry_row(100_000, 95_000, 2_000);
        let c = commit_row(&input, EffectiveMode::Acting, &RuleSwitches::ALL_ON);
        assert!(c.attack_contracts, "CR-C commits the governed attack slice");
        assert_eq!(c.terminal, Some(LaneSlot::Verifier), "CR-T: the verifier");
        assert!(c.release_hand_on, "CR-H hands releases on to the terminal");
        for cap in &mut input.capabilities {
            if cap.id.starts_with("attack.") {
                cap.admission = AdmissionState::Decline("none".to_owned());
            }
        }
        let c = commit_row(&input, EffectiveMode::Acting, &RuleSwitches::ALL_ON);
        assert!(!c.attack_contracts);
        assert_eq!(
            rule(&c, "CR-C").blocked_by.as_deref(),
            Some("attack_slice_ungoverned")
        );
        let c = commit_row(&input, EffectiveMode::Acting, &switches_without("CR-T"));
        assert_eq!(c.terminal, None);
        assert!(
            !c.release_hand_on,
            "no terminal: releases return to the pool"
        );
    }

    #[test]
    fn router_r2_backoffs_never_shorten_todays_reserves() {
        let mut rng = Rng(0x5eed_2202);
        for _ in 0..500 {
            let official = rng.below(2_000_000);
            let internal = rng.below(official.saturating_add(1));
            let input = registry_row(official, internal, rng.below(5_000));
            let c = commit_row(&input, EffectiveMode::Acting, &RuleSwitches::ALL_ON);
            for g in &c.governance {
                let expected = match g.slot {
                    LaneSlot::MarginRowConcurrent | LaneSlot::MarginRowInline => {
                        derive::MARGIN_ROW_TAIL_MS
                    }
                    _ => official - internal,
                };
                assert_eq!(g.backoff_ms, expected, "{g:?}");
            }
            // The verifier's cap is exactly the internal tier's end: the
            // router never plans it below the publication floor it keeps today.
            let anchor = Instant::now();
            let frame = Frame::new(
                anchor,
                anchor + Duration::from_millis(official),
                c.backoffs_ms().map(Duration::from_millis),
                Duration::from_millis(input.budget.join_margin_ms),
            );
            assert_eq!(
                frame.cap_ns(LaneSlot::Verifier),
                internal * 1_000_000,
                "verifier cap == internal tier"
            );
        }
    }

    #[test]
    fn router_r2_publish_grace_is_bounded_below_the_join_margin() {
        for join in 0..=10_000u64 {
            match derive::publish_grace_ms(join) {
                Some(grace) => {
                    assert!(grace > 0);
                    assert!(grace < derive::PUBLISH_GRACE_CEILING_MS, "{join}: {grace}");
                    assert!(grace < join, "{join}: {grace}");
                    assert!(
                        join - grace >= derive::WRITE_RESERVE_MS,
                        "{join}: the write reserve survives"
                    );
                }
                None => assert!(join <= derive::WRITE_RESERVE_MS, "{join}"),
            }
        }
        assert_eq!(
            derive::publish_grace_ms(2_000),
            Some(1_500),
            "today's 2 s join"
        );
        let input = registry_row(100_000, 95_000, 2_000);
        let c = commit_row(&input, EffectiveMode::Acting, &RuleSwitches::ALL_ON);
        assert_eq!(c.grace_ms, Some(1_500));
        // The grace arms the authority inside its own bounds.
        let anchor = Instant::now();
        let frame = Frame::new(
            anchor,
            anchor + Duration::from_secs(100),
            c.backoffs_ms().map(Duration::from_millis),
            Duration::from_millis(2_000),
        );
        let authority = Authority::new();
        authority
            .arm(
                frame,
                c.governed_flags(),
                c.terminal,
                c.grace_ms.map(Duration::from_millis),
            )
            .expect("the committed grace arms");
        assert_eq!(
            authority.publish_grace(),
            Some(Duration::from_millis(1_500))
        );
        for slot in LaneSlot::ALL {
            assert_eq!(authority.governs(slot), c.governed(slot));
        }
        // A join margin that leaves no room past the write reserve: no grace.
        let tight = registry_row(100_000, 95_000, 400);
        let c = commit_row(&tight, EffectiveMode::Acting, &RuleSwitches::ALL_ON);
        assert_eq!(c.grace_ms, None);
        assert_eq!(
            rule(&c, "CR-R").blocked_by.as_deref(),
            Some("join_margin_too_small")
        );
        // Ungoverned margin-row lanes get no grace (a grace without an
        // extended collector would publish into a join that stopped waiting).
        let mut open = registry_row(100_000, 95_000, 2_000);
        for cap in &mut open.capabilities {
            if cap.id == "margin-row" {
                cap.admission = AdmissionState::Decline("no-head".to_owned());
            }
        }
        let c = commit_row(&open, EffectiveMode::Acting, &RuleSwitches::ALL_ON);
        assert_eq!(c.grace_ms, None);
    }

    #[test]
    fn router_r2_derived_caps_never_go_below_today() {
        let mut last_depth = 0u64;
        let mut last_passes = 0u64;
        for window_s in (0..=4_000u64).step_by(7) {
            let window = window_s * 1_000;
            let depth = derive::max_depth(window, TODAY_MAX_DEPTH);
            let passes = derive::clause_passes(window);
            assert!((TODAY_MAX_DEPTH..=derive::DEPTH_CEILING).contains(&depth));
            assert!((1..=derive::PASS_CEILING).contains(&passes));
            assert!(
                depth >= last_depth && passes >= last_passes,
                "monotone in the window"
            );
            last_depth = depth;
            last_passes = passes;
        }
        assert_eq!(
            derive::max_depth(u64::MAX, TODAY_MAX_DEPTH),
            derive::DEPTH_CEILING
        );
        assert_eq!(derive::max_depth(0, TODAY_MAX_DEPTH), TODAY_MAX_DEPTH);
        // The 600 s ml4acopf row: the derived cap sits above the depth that
        // converted it (100 000, measured) and far above today's 100.
        assert!(derive::max_depth(593_000, TODAY_MAX_DEPTH) >= 100_000);
        // The depth lift needs a resident-queue byte budget inside the lease.
        let lease = 105u64 << 30;
        assert_eq!(derive::depth_lift_gate(0, lease), DepthGate::NoQueueBudget);
        assert_eq!(derive::depth_lift_gate(2 << 30, lease), DepthGate::Admit);
        assert_eq!(
            derive::depth_lift_gate(lease, lease),
            DepthGate::BudgetPastLease
        );
        assert_eq!(derive::depth_lift_gate(2 << 30, 0), DepthGate::LeaseUnknown);
        let input = registry_row(600_000, 570_000, 2_000);
        let c = commit_row(&input, EffectiveMode::Acting, &RuleSwitches::ALL_ON);
        assert_eq!(c.derived.lease_bytes, host().envelope_bytes());
        assert!(c.derived.max_depth.expect("CR-K acts") >= 100_000);
        assert!(c.derived.warmup_hand_on, "CR-H hands the warmup's time on");
        let off = commit_row(&input, EffectiveMode::Acting, &switches_without("CR-K"));
        assert_eq!(off.derived.clause_passes, None);
        assert_eq!(off.derived.max_depth, None);
    }

    #[test]
    fn router_r2_arithmetic_never_panics_on_extremes() {
        for official in [0u64, 1, 30_000, u64::MAX] {
            for internal in [0u64, 1, official, u64::MAX] {
                for join in [0u64, 1, 2_000, u64::MAX] {
                    let input = registry_row(official, internal, join);
                    for mode in MODES {
                        let c = commit_row(&input, mode, &RuleSwitches::ALL_ON);
                        let _ = c.backoffs_ms();
                        let _ = c.governed_flags();
                    }
                }
            }
        }
        // An empty capability list governs nothing and still commits.
        let mut empty = registry_row(100_000, 95_000, 2_000);
        empty.capabilities.clear();
        let c = commit_row(&empty, EffectiveMode::Acting, &RuleSwitches::ALL_ON);
        assert!(c.governance.iter().all(|g| !g.governed));
    }
}

/// #router-r3 (ROUTER_PROGRAM §10 R3): acting set 2, the O-STOP rules.
mod r3 {
    use std::time::{Duration, Instant};

    use super::super::acting::{self, Commitments, RuleSwitches};
    use super::super::ostop::{
        self, attack_flat, OptimizerSwitches, SlopeVerdict, StopCommitments, R3_RULES,
    };
    use super::super::plan::{
        self, AdmissionState, BudgetView, CapClass, LedgerState, PolicyInput,
    };
    use super::super::{EffectiveMode, PRIORS_VERSION, ROUTER_SCHEMA};
    use super::{cap, globals, guard, host, random_input, Rng};
    use crate::router::decl::{LaneRole, ProgressKind};
    use crate::router::facts::InstanceFacts;
    use crate::stop_authority::{self, StopDirectives};

    const MODES: [EffectiveMode; 3] = [
        EffectiveMode::Off,
        EffectiveMode::Shadow,
        EffectiveMode::Acting,
    ];

    fn row(official_ms: u64) -> PolicyInput {
        let lane = |id: &str, role: LaneRole| {
            cap(
                id,
                CapClass::Lane(role),
                LedgerState::SealedAsShipped,
                ProgressKind::Opaque,
                None,
                AdmissionState::Unknown("inline-admission".to_owned()),
            )
        };
        let phase = |id: &str| {
            cap(
                id,
                CapClass::Phase {
                    lane: "margin-row".to_owned(),
                },
                LedgerState::SealedAsShipped,
                ProgressKind::Yield,
                None,
                AdmissionState::Unknown("parent-unknown".to_owned()),
            )
        };
        let prover = LaneRole::Prover { terminal_ok: true };
        PolicyInput {
            schema: ROUTER_SCHEMA.to_owned(),
            facts: InstanceFacts::known_empty(),
            host: host(),
            budget: BudgetView {
                official_ms,
                internal_ms: official_ms.saturating_sub(5_000),
                join_margin_ms: 2_000,
            },
            capabilities: vec![
                lane("attack.upfront_apgd", LaneRole::Attack),
                lane("attack.bnn_lp", LaneRole::Attack),
                lane("attack.bnn_ste", LaneRole::Attack),
                lane("relu_split.graph", prover),
                lane("margin-row", prover),
                // #router-r4: the phases the O-STOP rules end early, declared
                // with safe points as the registry declares them (a stop
                // rule acts only on a capability that declares one).
                phase("margin-row.alpha"),
                phase("margin-row.backward_interm"),
            ],
            pins: Vec::new(),
            priors_version: PRIORS_VERSION.to_owned(),
            registry_version: "0123abcd".to_owned(),
            priors: None,
        }
    }

    fn r2(input: &PolicyInput, mode: EffectiveMode, switches: &RuleSwitches) -> Commitments {
        acting::commit(&plan::plan(input), input, mode, switches)
    }

    fn commit(
        input: &PolicyInput,
        mode: EffectiveMode,
        switches: &OptimizerSwitches,
        router: Option<&Commitments>,
    ) -> StopCommitments {
        ostop::commit(&plan::plan(input), input, mode, switches, router)
    }

    fn rule<'a>(c: &'a StopCommitments, id: &str) -> &'a acting::RuleAct {
        c.rules
            .iter()
            .find(|act| act.rule == id)
            .unwrap_or_else(|| panic!("rule {id} missing from {:?}", c.rules))
    }

    fn without(id: &str) -> OptimizerSwitches {
        let mut s = OptimizerSwitches::ALL_ON;
        match id {
            "rule.governor" => s.governor = false,
            "rule.ia_yield" => s.ia_yield = false,
            "rule.attack_defer" => s.attack_defer = false,
            "rule.root_admission" => s.root_admission = false,
            other => panic!("unknown rule {other}"),
        }
        s
    }

    #[test]
    fn router_r3_stop_commitments_are_pure_and_deterministic() {
        let mut rng = Rng(0x5eed_3301);
        for _ in 0..400 {
            let input = random_input(&mut rng);
            let plan = plan::plan(&input);
            let router =
                acting::commit(&plan, &input, EffectiveMode::Acting, &RuleSwitches::ALL_ON);
            for mode in MODES {
                let switches = OptimizerSwitches {
                    governor: rng.coin(),
                    ia_yield: rng.coin(),
                    attack_defer: rng.coin(),
                    root_admission: rng.coin(),
                };
                let with = rng.coin().then_some(&router);
                let a = ostop::commit(&plan, &input, mode, &switches, with);
                let b = ostop::commit(&plan, &input, mode, &switches, with);
                assert_eq!(a, b, "commit is a pure function of its input");
                assert_eq!(a.sha, b.sha);
                assert_eq!(a.sha.len(), 64);
                assert_eq!(a.rules.len(), R3_RULES.len(), "every rule decides");
                for (act, id) in a.rules.iter().zip(R3_RULES) {
                    assert_eq!(act.rule, id.as_str(), "rules are reported in R3 order");
                    assert_eq!(act.acts, act.blocked_by.is_none(), "{act:?}");
                }
                // The directives are exactly the acting rules.
                assert_eq!(
                    a.directives.governor.is_some(),
                    a.acts(ostop::RULE_GOVERNOR)
                );
                assert_eq!(
                    a.directives.ia_yield.is_some(),
                    a.acts(ostop::RULE_IA_YIELD)
                );
                assert_eq!(
                    a.directives.attack_defer.is_some(),
                    a.acts(ostop::RULE_ATTACK_DEFER)
                );
                assert_eq!(
                    a.directives.root_admission,
                    a.acts(ostop::RULE_ROOT_ADMISSION)
                );
            }
        }
    }

    #[test]
    fn router_r3_nothing_stops_unless_the_optimizer_acts() {
        let input = row(100_000);
        let router = r2(&input, EffectiveMode::Acting, &RuleSwitches::ALL_ON);
        for mode in [EffectiveMode::Off, EffectiveMode::Shadow] {
            let c = commit(&input, mode, &OptimizerSwitches::ALL_ON, Some(&router));
            assert!(!c.acting);
            assert_eq!(c.directives, StopDirectives::NONE, "{mode:?} arms nothing");
            for act in &c.rules {
                assert!(!act.acts, "{mode:?}: {act:?}");
                assert_eq!(
                    act.blocked_by.as_deref(),
                    Some(format!("mode:{}", mode.token()).as_str())
                );
            }
        }
        let on = commit(
            &input,
            EffectiveMode::Acting,
            &OptimizerSwitches::ALL_ON,
            Some(&router),
        );
        assert!(on.acting, "positive control: `on` acts");
        assert!(R3_RULES.iter().all(|id| on.acts(*id)), "{:?}", on.rules);
        assert!(on.directives.any());
    }

    /// The measured constants ship: governor 0.15 after 5 samples (the
    /// moat-cleared artifact), interm 0.15 after 2 layers, CR-ATK over half
    /// the slice with a 5 s floor (the late attack's own minimum).
    #[test]
    fn router_r3_directives_carry_the_measured_constants() {
        let input = row(100_000);
        let router = r2(&input, EffectiveMode::Acting, &RuleSwitches::ALL_ON);
        let on = commit(
            &input,
            EffectiveMode::Acting,
            &OptimizerSwitches::ALL_ON,
            Some(&router),
        );
        let gov = on.directives.governor.expect("governor");
        assert!((gov.ratio - 0.15).abs() < f64::EPSILON);
        assert_eq!(gov.min_samples, 5);
        let ia = on.directives.ia_yield.expect("ia_yield");
        assert!((ia.ratio - 0.15).abs() < f64::EPSILON);
        assert_eq!(ia.min_samples, 2);
        let defer = on.directives.attack_defer.expect("attack_defer");
        assert!((defer.window_frac - 0.5).abs() < f64::EPSILON);
        assert_eq!(defer.min_defer_ms, 5_000);
        assert!(gov.is_valid() && ia.is_valid() && defer.is_valid());
        assert!(on.directives.root_admission);
    }

    #[test]
    fn router_r3_each_rule_key_ablates_its_own_rule() {
        let input = row(100_000);
        let router = r2(&input, EffectiveMode::Acting, &RuleSwitches::ALL_ON);
        for id in R3_RULES {
            let c = commit(
                &input,
                EffectiveMode::Acting,
                &without(id.as_str()),
                Some(&router),
            );
            let act = rule(&c, id.as_str());
            assert!(!act.acts, "{id:?}");
            assert_eq!(act.blocked_by.as_deref(), Some("key"), "{id:?}");
            for other in R3_RULES.iter().filter(|o| **o != id) {
                assert!(
                    c.acts(*other),
                    "ablating {id:?} must leave {other:?} acting"
                );
            }
        }
    }

    /// `rule.ia_yield` is eligible ONLY once CR-1 makes the ascent read a
    /// contract; CR-ATK rides on CR-1 and the upfront slot's governance. The
    /// governor and root admission need no cell.
    #[test]
    fn router_r3_ia_yield_and_attack_defer_ride_on_cr1() {
        let input = row(100_000);
        for router in [
            None,
            Some(r2(&input, EffectiveMode::Shadow, &RuleSwitches::ALL_ON)),
            Some(r2(
                &input,
                EffectiveMode::Acting,
                &RuleSwitches {
                    cap_authority: false,
                    ..RuleSwitches::ALL_ON
                },
            )),
        ] {
            let c = commit(
                &input,
                EffectiveMode::Acting,
                &OptimizerSwitches::ALL_ON,
                router.as_ref(),
            );
            for id in ["rule.ia_yield", "rule.attack_defer"] {
                let act = rule(&c, id);
                assert!(!act.acts, "{id} rides on CR-1: {act:?}");
                assert_eq!(act.blocked_by.as_deref(), Some("CR-1"), "{id}");
            }
            assert!(c.directives.ia_yield.is_none() && c.directives.attack_defer.is_none());
            assert!(c.acts(ostop::RULE_GOVERNOR));
            assert!(c.acts(ostop::RULE_ROOT_ADMISSION));
        }
    }

    #[test]
    fn router_r3_attack_defer_needs_the_upfront_slot_and_a_long_row() {
        // CR-A leaves the upfront slot ungoverned when its lane declined.
        let mut declined = row(100_000);
        declined.capabilities[0].admission = AdmissionState::Decline("not-admitted".to_owned());
        let router = r2(&declined, EffectiveMode::Acting, &RuleSwitches::ALL_ON);
        let c = commit(
            &declined,
            EffectiveMode::Acting,
            &OptimizerSwitches::ALL_ON,
            Some(&router),
        );
        assert_eq!(
            rule(&c, "rule.attack_defer").blocked_by.as_deref(),
            Some("attack_upfront_ungoverned")
        );
        // CR-S: a short row takes no defer.
        let short = row(20_000);
        let router = r2(&short, EffectiveMode::Acting, &RuleSwitches::ALL_ON);
        let c = commit(
            &short,
            EffectiveMode::Acting,
            &OptimizerSwitches::ALL_ON,
            Some(&router),
        );
        assert_eq!(
            rule(&c, "rule.attack_defer").blocked_by.as_deref(),
            Some("short_row")
        );
        assert!(c.directives.attack_defer.is_none());
    }

    fn stream(points: &[(f64, f64)]) -> Vec<(f64, f64)> {
        // Restart-boundary samples carry the RUNNING max.
        let mut best = f64::NEG_INFINITY;
        points
            .iter()
            .map(|&(t, m)| {
                best = best.max(m);
                (t, best)
            })
            .collect()
    }

    /// Synthetic progress streams through CR-ATK's slope predicate: the rate
    /// decides, never the level.
    #[test]
    fn router_r3_attack_flat_reads_the_rate_not_the_level() {
        const NOISE: f64 = 1e-4;
        // metaroom spec_idx_28 shape (GT unsat): plateaued at its sup -0.033.
        let plateau = stream(&(0..=16).map(|s| (f64::from(s), -0.033)).collect::<Vec<_>>());
        assert!(attack_flat(&plateau, 30.0, 0.5, NOISE).is_flat());
        // ... but not before a full half-slice window has elapsed.
        assert_eq!(
            attack_flat(&plateau[..10], 30.0, 0.5, NOISE),
            SlopeVerdict::Unknown("window_not_elapsed")
        );
        // spec_idx_129 shape (GT sat): still climbing at -0.083 — CLOSER to
        // nothing, FARTHER than the unsat sup, and it must keep its budget.
        let climbing = stream(
            &(0..=29)
                .map(|s| (f64::from(s), -2.0 + 0.0665 * f64::from(s)))
                .collect::<Vec<_>>(),
        );
        assert_eq!(
            attack_flat(&climbing, 30.0, 0.5, NOISE),
            SlopeVerdict::Climbing
        );
        // Sub-noise creep is no progress.
        let creep = stream(
            &(0..=16)
                .map(|s| (f64::from(s), -0.033 + 1e-6 * f64::from(s)))
                .collect::<Vec<_>>(),
        );
        assert!(attack_flat(&creep, 30.0, 0.5, NOISE).is_flat());
        // Above-noise progress inside the window keeps the attack running.
        let late_jump = stream(&[(0.0, -0.5), (8.0, -0.5), (15.0, -0.5), (16.0, -0.3)]);
        assert_eq!(
            attack_flat(&late_jump, 30.0, 0.5, NOISE),
            SlopeVerdict::Climbing
        );
        // A point at the violation surface: the candidate path owns it.
        let hit = stream(&[(0.0, -0.5), (16.0, 0.0)]);
        assert_eq!(attack_flat(&hit, 30.0, 0.5, NOISE), SlopeVerdict::Climbing);
        // No sample at or before the window start: the window is not spanned.
        let sparse = stream(&[(16.0, -0.4), (17.0, -0.4)]);
        assert_eq!(
            attack_flat(&sparse, 30.0, 0.5, NOISE),
            SlopeVerdict::Unknown("window_not_spanned")
        );
        // A window fraction below one half is clamped UP to one half.
        let early = stream(&(0..=10).map(|s| (f64::from(s), -0.033)).collect::<Vec<_>>());
        assert_eq!(
            attack_flat(&early, 30.0, 0.1, NOISE),
            SlopeVerdict::Unknown("window_not_elapsed"),
            "W >= 0.5 of the slice, whatever is asked"
        );
        // Non-finite samples carry no signal; none at all is Unknown.
        assert_eq!(
            attack_flat(
                &[(f64::NAN, -0.1), (1.0, f64::NEG_INFINITY)],
                30.0,
                0.5,
                NOISE
            ),
            SlopeVerdict::Unknown("no_finite_sample")
        );
        assert_eq!(
            attack_flat(&[], 30.0, 0.5, NOISE),
            SlopeVerdict::Unknown("no_finite_sample")
        );
        // A non-finite noise floor only makes a defer HARDER (any rise counts).
        let tiny_rise = stream(&[(0.0, -0.5), (16.0, -0.5 + 1e-9)]);
        assert_eq!(
            attack_flat(&tiny_rise, 30.0, 0.5, f64::NAN),
            SlopeVerdict::Climbing
        );
    }

    #[test]
    fn router_r3_arithmetic_never_panics_on_extremes() {
        for official in [0u64, 1, 30_000, 100_000, u64::MAX] {
            let input = row(official);
            for mode in MODES {
                let router = r2(&input, mode, &RuleSwitches::ALL_ON);
                let _ = commit(&input, mode, &OptimizerSwitches::ALL_ON, Some(&router));
            }
        }
        for slice in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::MAX, 1e-300] {
            for frac in [0.0, 0.5, 2.0, f64::NAN, f64::INFINITY] {
                for eps in [0.0, -1.0, f64::NAN, f64::INFINITY] {
                    let pts = [
                        (0.0, -1.0),
                        (f64::MAX, -0.5),
                        (f64::INFINITY, f64::MAX),
                        (1.0, f64::MIN),
                    ];
                    let _ = attack_flat(&pts, slice, frac, eps);
                }
            }
        }
    }

    /// #router-r3 review: a running best AT or ABOVE the violation surface is
    /// never judged flat, however long it has sat there. The builder's `hit`
    /// case above rises by 0.5 inside the window, so it reads `Climbing`
    /// without this guard: only a PLATEAU at the surface exercises it. Such a
    /// point is the candidate path's (an ORT-unconfirmed margin >= 0, a
    /// pre-Softmax score, a non-strict boundary), never a stop rule's — a
    /// defer there would end the attack on its most promising iterate.
    #[test]
    fn router_r3_review_attack_flat_never_defers_at_the_violation_surface() {
        const NOISE: f64 = 1e-4;
        for level in [0.0, 1e-12, 0.25] {
            let surface = stream(&(0..=16).map(|s| (f64::from(s), level)).collect::<Vec<_>>());
            assert_eq!(
                attack_flat(&surface, 30.0, 0.5, NOISE),
                SlopeVerdict::Climbing,
                "a plateau at {level} is the candidate path's, never a defer"
            );
        }
        // Positive control: the same plateau just below the surface IS flat.
        let below = stream(&(0..=16).map(|s| (f64::from(s), -1e-3)).collect::<Vec<_>>());
        assert!(attack_flat(&below, 30.0, 0.5, NOISE).is_flat());
    }

    /// A contained router fault disarms the STOP authority too: every phase
    /// reads `None` and keeps today's rule.
    #[test]
    fn router_r3_fault_disarms_the_stop_authority() {
        let _serial = globals();
        guard::reset();
        let stop = stop_authority::global();
        stop.arm(StopDirectives {
            governor: Some(ostop::GOVERNOR_RULE),
            ia_yield: Some(ostop::IA_YIELD_RULE),
            attack_defer: Some(ostop::ATTACK_DEFER_RULE),
            root_admission: true,
        })
        .expect("arm");
        assert!(stop.governor().is_some(), "positive control");
        let fault = guard::fail_open("injected-r3", || -> u32 {
            panic!("router-r3 injected panic")
        })
        .expect_err("the injected panic is contained");
        assert!(!fault.skipped);
        assert!(
            !stop.is_armed(),
            "a contained fault disarms the stop authority"
        );
        assert_eq!(stop.governor(), None);
        assert_eq!(stop.ia_yield(), None);
        assert_eq!(stop.attack_defer(), None);
        assert!(!stop.root_admission());
        stop.credit_ascent(Duration::from_secs(5));
        assert_eq!(stop.take_ascent_credit(), Duration::ZERO);
        guard::reset();
        super::bus::reset();
    }

    /// Monotone time conservation with the R2 cells: an O-STOP stop inside a
    /// lane without safe points never writes that lane's cell; the freed time
    /// reaches the next phase inside the cell's window, and the sum of freed
    /// time never exceeds the window.
    #[test]
    fn router_r3_stops_conserve_time_inside_the_r2_cells() {
        use crate::deadline_authority::{Authority, Frame, LaneSlot, SLOTS};
        use crate::stop_authority::{StopAuthority, StopSite, YieldEvent};
        let cells = Authority::new();
        let anchor = Instant::now();
        let frame = Frame::new(
            anchor,
            anchor + Duration::from_secs(100),
            [
                Duration::from_secs(5),
                Duration::from_secs(5),
                Duration::from_secs(5),
                Duration::from_secs(5),
                Duration::from_secs(3),
                Duration::from_secs(3),
            ],
            Duration::from_secs(2),
        );
        cells
            .arm(frame, [true; SLOTS], Some(LaneSlot::Verifier), None)
            .expect("arm cells");
        let lane_deadline = anchor + Duration::from_secs(97);
        cells
            .claim_deadline(LaneSlot::MarginRowInline, lane_deadline, anchor)
            .expect("claim");
        let held = cells.deadline(LaneSlot::MarginRowInline);
        let stop = StopAuthority::new();
        stop.arm(StopDirectives {
            governor: Some(ostop::GOVERNOR_RULE),
            ..StopDirectives::NONE
        })
        .expect("arm stops");
        let mut rng = Rng(0x5eed_3302);
        let mut clock = Duration::from_secs(10);
        let mut freed_total = 0u64;
        for _ in 0..50 {
            let budget = rng.below(20_000) + 1;
            let used = rng.below(budget + 1);
            let freed = budget - used;
            clock += Duration::from_millis(used);
            stop.record_yield(YieldEvent {
                site: StopSite::AlphaAscent,
                tag: "[alpha-opt]".to_owned(),
                unit: rng.below(20),
                budget_ms: budget,
                used_ms: used,
                freed_ms: freed,
                rate: 0.0,
                best: 1.0,
                floor: 0.15,
            });
            let now = anchor + clock;
            let window = lane_deadline.saturating_duration_since(now);
            for hand in stop.downstream_start("margin-row.tree", Some(window)) {
                assert!(hand.credited_ms <= hand.freed_ms);
                assert!(hand.credited_ms <= u64::try_from(window.as_millis()).unwrap_or(u64::MAX));
                assert_eq!(hand.realized, hand.credited_ms == hand.freed_ms);
                freed_total = freed_total.saturating_add(hand.credited_ms);
            }
            // The lane's cell never moves: O-STOP writes no R2 cell.
            assert_eq!(cells.deadline(LaneSlot::MarginRowInline), held);
            if clock > Duration::from_secs(90) {
                break;
            }
        }
        let snap = stop.snapshot();
        let credited: u64 = snap.hand_ons.iter().map(|h| h.credited_ms).sum();
        assert_eq!(credited, freed_total);
        assert!(snap.hand_ons.iter().all(|h| h.credited_ms <= h.freed_ms));
        assert_eq!(
            cells.snapshot().hand_ons.len(),
            0,
            "no lane released anything"
        );
    }
}

// ============================================================================
// #router-r4 — acting set 3 (ROUTER_PROGRAM §10 R4): the memory lease (CR-L),
// enforceable safe points, and the FAR hand-on (CR-F).
// ============================================================================

mod r4 {
    use std::time::{Duration, Instant};

    use super::super::acting::{self, Commitments, RuleSwitches};
    use super::super::bus::ProgressSample;
    use super::super::checkpoint::{self, Amendment, CheckpointId, FarBlock};
    use super::super::decl::{CapId, LaneRole, ProgressKind, SafePoints};
    use super::super::facts::{DeviceQual, HostClass, HostEnvelope, InstanceFacts};
    use super::super::lease::{
        self, decide, LeaseDecision, LeaseHolder, LeaseNeed, OPTIONAL_HEAVY_LEASE_BYTES,
    };
    use super::super::ostop::{self, OptimizerSwitches};
    use super::super::plan::{
        self, AdmissionState, BudgetView, CapClass, LedgerState, Plan, PolicyInput,
    };
    use super::super::project::{self, FarClass};
    use super::super::r4::{self, FarCommit, FarRoute, R4Commitments, R4Switches, R4_RULES};
    use super::super::safepoint::{self, At, Ruling};
    use super::super::{EffectiveMode, PRIORS_VERSION, ROUTER_SCHEMA};
    use super::{cap, collapsing, fixtures, globals, guard, host, random_input, Rng, FIXTURE_CAP};
    use crate::deadline_authority::{Authority, Frame, LaneSlot, SLOTS};
    use crate::far_authority::{FarAction, FarAuthority};
    use crate::memory_lease::LeaseAuthority;
    use crate::stop_authority::StopSite;

    const GIB: u64 = 1 << 30;

    const MODES: [EffectiveMode; 3] = [
        EffectiveMode::Off,
        EffectiveMode::Shadow,
        EffectiveMode::Acting,
    ];

    fn prover(
        id: &str,
        ledger: LedgerState,
        progress: ProgressKind,
        admission: AdmissionState,
    ) -> plan::CapInput {
        cap(
            id,
            CapClass::Lane(LaneRole::Prover { terminal_ok: true }),
            ledger,
            progress,
            None,
            admission,
        )
    }

    /// The registry-shaped row R2 governs (every slot's capability, inline
    /// admissions, sealed-as-shipped), plus the given extra capabilities.
    fn row(official_ms: u64, extra: Vec<plan::CapInput>) -> PolicyInput {
        let lane = |id: &str, role: LaneRole| {
            cap(
                id,
                CapClass::Lane(role),
                LedgerState::SealedAsShipped,
                ProgressKind::Opaque,
                None,
                AdmissionState::Unknown("inline-admission".to_owned()),
            )
        };
        let p = LaneRole::Prover { terminal_ok: true };
        let mut capabilities = vec![
            lane("attack.upfront_apgd", LaneRole::Attack),
            lane("attack.bnn_lp", LaneRole::Attack),
            lane("attack.bnn_ste", LaneRole::Attack),
            lane("mip.ay", p),
            lane("input_split", p),
            lane("relu_split.graph", p),
            lane("margin-row", p),
        ];
        capabilities.extend(extra);
        PolicyInput {
            schema: ROUTER_SCHEMA.to_owned(),
            facts: InstanceFacts::known_empty(),
            host: host(),
            budget: BudgetView {
                official_ms,
                internal_ms: official_ms.saturating_sub(5_000),
                join_margin_ms: 2_000,
            },
            capabilities,
            pins: Vec::new(),
            priors_version: PRIORS_VERSION.to_owned(),
            registry_version: "0123abcd".to_owned(),
            priors: None,
        }
    }

    fn commits(
        input: &PolicyInput,
        mode: EffectiveMode,
        r2_switches: &RuleSwitches,
        switches: &R4Switches,
    ) -> (Plan, Commitments, R4Commitments) {
        let p = plan::plan(input);
        let r2 = acting::commit(&p, input, mode, r2_switches);
        let c = r4::commit(&p, input, mode, switches, &r2);
        (p, r2, c)
    }

    fn rule<'a>(c: &'a R4Commitments, id: &str) -> &'a acting::RuleAct {
        c.rules
            .iter()
            .find(|act| act.rule == id)
            .unwrap_or_else(|| panic!("rule {id} missing from {:?}", c.rules))
    }

    // ------------------------------------------------------------------
    // Safe points.
    // ------------------------------------------------------------------

    /// THE R4 safe-point gate (§3.2 C8, §6.1 "cancellation"): the router may
    /// suspend or shorten a capability ONLY at a declared safe point, and
    /// `None` means NEVER. Checked at every layer that can end a lane early:
    /// the ruling table itself, CR-F's pure recipient search, the live FAR
    /// authority, and the O-STOP commitments.
    #[test]
    fn router_suspends_only_at_safe_points() {
        // 1. The ruling table.
        for at in [At::Start, At::Unit(0), At::Unit(u64::MAX), At::Inside] {
            assert_eq!(
                safepoint::may_suspend(SafePoints::None, at),
                Ruling::Refused(safepoint::REFUSED_NO_SAFE_POINTS),
                "None is never suspended ({at:?})"
            );
        }
        assert!(safepoint::may_suspend(SafePoints::BetweenUnits, At::Start).allowed());
        assert!(safepoint::may_suspend(SafePoints::BetweenUnits, At::Unit(7)).allowed());
        assert_eq!(
            safepoint::may_suspend(SafePoints::BetweenUnits, At::Inside),
            Ruling::Refused(safepoint::REFUSED_NOT_A_SAFE_POINT)
        );
        // A raise is never a shortening; a shortening follows may_suspend.
        assert!(safepoint::may_shorten(SafePoints::None, At::Start, 10, 10).allowed());
        assert!(safepoint::may_shorten(SafePoints::None, At::Start, 10, 11).allowed());
        assert!(!safepoint::may_shorten(SafePoints::None, At::Start, 10, 9).allowed());
        assert!(safepoint::may_shorten(SafePoints::BetweenUnits, At::Start, 10, 9).allowed());
        assert_eq!(safepoint::declared_or_never(None), SafePoints::None);
        assert_eq!(
            safepoint::strictest(SafePoints::BetweenUnits, SafePoints::None),
            SafePoints::None
        );

        // 2. CR-F's pure decision: a FAR source WITHOUT safe points never
        // hands on, even with a strictly sealed recipient; the same source
        // with safe points does (positive control).
        let mut input = super::hand_on_input();
        let samples = collapsing("far", 10.0);
        let p = plan::plan(&input);
        assert!(
            matches!(
                checkpoint::checkpoint(&p, &samples, CheckpointId::C2).as_slice(),
                [Amendment::HandOn { .. }]
            ),
            "control: a FAR lane with safe points hands on"
        );
        input.capabilities[0].safe_points = SafePoints::None;
        let p = plan::plan(&input);
        assert_eq!(
            checkpoint::checkpoint(&p, &samples, CheckpointId::C2).len(),
            0
        );
        assert_eq!(
            checkpoint::far_recipient(&p, "far").map(|c| c.id.clone()),
            Err(FarBlock::NoSafePoints)
        );
        let d = checkpoint::far_decision(&p, &samples, "far", CheckpointId::C2);
        assert_eq!(d.class, FarClass::Far, "still classified (and logged)");
        assert!(!d.act);
        assert_eq!(d.blocked_by, Some(FarBlock::NoSafePoints));

        // 3. The live authority: only at a unit boundary of a capability that
        // declares safe points.
        let input = super::hand_on_input();
        let p = plan::plan(&input);
        let far = FarAuthority::new();
        let cells = Authority::new();
        far.arm(
            p,
            FarCommit {
                routes: Vec::new(),
                observe_only: true,
            },
            Instant::now(),
        );
        let sample = ProgressSample::frontier(CapId("far"), 0, 1, 0, 1);
        assert_eq!(
            far.decide_at(sample, At::Inside, 0.0, &cells, Instant::now()),
            FarAction::Continue
        );
        assert_eq!(
            far.decide_at(
                ProgressSample::frontier(CapId("undeclared"), 0, 1, 0, 1),
                At::Unit(0),
                0.0,
                &cells,
                Instant::now()
            ),
            FarAction::Continue
        );
        let snap = far.snapshot();
        assert_eq!(snap.refusals.len(), 2, "both requests refused and counted");
        assert_eq!(snap.refusals[0].reason, safepoint::REFUSED_NOT_A_SAFE_POINT);
        assert_eq!(snap.refusals[1].reason, safepoint::REFUSED_NO_SAFE_POINTS);
        let _ = far.decide_at(sample, At::Unit(0), 0.0, &cells, Instant::now());
        assert_eq!(
            far.snapshot().refusals.len(),
            2,
            "control: a unit boundary of a declared capability is a safe point"
        );

        // 4. The O-STOP rules that END a phase early act only on a capability
        // that declares safe points (`blocked_by=safe_points`); root admission
        // (skipping BEFORE a phase starts) is admission, not a suspension.
        let phase = |id: &str, points: SafePoints| {
            let mut c = cap(
                id,
                CapClass::Phase {
                    lane: "margin-row".to_owned(),
                },
                LedgerState::SealedAsShipped,
                ProgressKind::Yield,
                None,
                AdmissionState::Unknown("parent-unknown".to_owned()),
            );
            c.safe_points = points;
            c
        };
        let stop_row = |points: SafePoints| {
            let mut input = row(
                100_000,
                vec![
                    phase(StopSite::AlphaAscent.token(), points),
                    phase(StopSite::IntermAlpha.token(), points),
                ],
            );
            for c in &mut input.capabilities {
                if c.id == StopSite::AttackUpfront.token() {
                    c.safe_points = points;
                }
            }
            input
        };
        for (points, acts) in [(SafePoints::BetweenUnits, true), (SafePoints::None, false)] {
            let input = stop_row(points);
            let p = plan::plan(&input);
            let r2 = acting::commit(&p, &input, EffectiveMode::Acting, &RuleSwitches::ALL_ON);
            let c = ostop::commit(
                &p,
                &input,
                EffectiveMode::Acting,
                &OptimizerSwitches::ALL_ON,
                Some(&r2),
            );
            for id in ["rule.governor", "rule.ia_yield", "rule.attack_defer"] {
                let act = c.rules.iter().find(|a| a.rule == id).expect("rule");
                assert_eq!(act.acts, acts, "{id} with {points:?}: {act:?}");
                if !acts {
                    assert_eq!(act.blocked_by.as_deref(), Some("safe_points"), "{id}");
                }
            }
            assert_eq!(c.directives.governor.is_some(), acts);
            assert_eq!(c.directives.ia_yield.is_some(), acts);
            assert_eq!(c.directives.attack_defer.is_some(), acts);
            assert!(c.directives.root_admission, "admission needs no safe point");
        }
        // Undeclared is never a licence.
        let mut undeclared = stop_row(SafePoints::BetweenUnits);
        undeclared
            .capabilities
            .retain(|c| c.id != StopSite::AlphaAscent.token());
        assert!(!ostop::site_suspendable(&undeclared, StopSite::AlphaAscent));
        assert!(ostop::site_suspendable(&undeclared, StopSite::IntermAlpha));
    }

    // ------------------------------------------------------------------
    // The lease.
    // ------------------------------------------------------------------

    #[test]
    fn router_r4_lease_arithmetic_grants_narrows_and_denies() {
        // decide(): the whole need, the narrowable floor, or nothing.
        assert_eq!(
            decide(20, LeaseNeed::whole(16)),
            LeaseDecision::Granted { bytes: 16 }
        );
        assert_eq!(
            decide(16, LeaseNeed::whole(16)),
            LeaseDecision::Granted { bytes: 16 }
        );
        assert_eq!(
            decide(15, LeaseNeed::whole(16)),
            LeaseDecision::Denied {
                available: 15,
                want: 16
            },
            "an indivisible need is never narrowed"
        );
        let narrowable = LeaseNeed {
            want_bytes: 16,
            min_bytes: 8,
        };
        assert!(narrowable.narrowable() && !LeaseNeed::whole(16).narrowable());
        assert_eq!(
            decide(10, narrowable),
            LeaseDecision::Narrowed {
                bytes: 10,
                want: 16
            }
        );
        assert_eq!(
            decide(7, narrowable),
            LeaseDecision::Denied {
                available: 7,
                want: 16
            }
        );
        assert!(!LeaseDecision::Denied {
            available: 0,
            want: 1
        }
        .admits());
        assert!(LeaseDecision::Narrowed { bytes: 1, want: 2 }.admits());
        assert_eq!(decide(0, LeaseNeed::whole(0)).bytes(), 0);
        // The plan: E_static from the DECLARED host, the terminal holds half,
        // every optional heavy lane asks for the measured 16 GiB boundary.
        for (class, envelope) in [
            (HostClass::Gb10, 105 * GIB),
            (HostClass::G5, 120 * GIB),
            (HostClass::M5, 240 * GIB),
        ] {
            let host = HostEnvelope::declared_target(class);
            let plan = lease::lease_plan(&host).expect("a known envelope");
            assert_eq!(plan.envelope_bytes, envelope, "{class:?}");
            assert_eq!(plan.terminal_bytes(), envelope / 2, "{class:?}");
            assert_eq!(plan.host, class);
            assert_eq!(plan.threads, host.threads, "width is a logged plan value");
            assert!(plan.declared);
            for holder in LeaseHolder::ALL {
                if holder == LeaseHolder::KfsbFanout {
                    // #kfsb-production: the one narrowable need — it wants
                    // the uniform heavy model and runs with as little as one
                    // simulation's share at its resident peak.
                    assert_eq!(plan.need(holder), lease::KFSB_FANOUT_LEASE_NEED);
                    assert!(plan.need(holder).narrowable());
                    assert_eq!(plan.need(holder).want_bytes, OPTIONAL_HEAVY_LEASE_BYTES);
                    assert_eq!(
                        plan.need(holder).min_bytes,
                        lease::DENSE_PAIR_PEAK_FACTOR * lease::KFSB_FANOUT_MIN_SHARE_BYTES
                    );
                } else if holder == LeaseHolder::AttackConcurrent {
                    // #attack-concurrent: the lane shares the model by
                    // reference, so its indivisible need is its own working
                    // set, not the graph-reload model.
                    assert_eq!(
                        plan.need(holder),
                        LeaseNeed::whole(lease::ATTACK_CONCURRENT_LEASE_BYTES)
                    );
                    assert!(!plan.need(holder).narrowable());
                    const {
                        assert!(lease::ATTACK_CONCURRENT_LEASE_BYTES < OPTIONAL_HEAVY_LEASE_BYTES);
                    }
                } else if holder != LeaseHolder::Terminal {
                    assert_eq!(
                        plan.need(holder),
                        LeaseNeed::whole(OPTIONAL_HEAVY_LEASE_BYTES),
                        "{holder:?}"
                    );
                }
                assert_eq!(LeaseHolder::from_token(holder.token()), Some(holder));
            }
        }
        // No envelope, no lease: today's live gate stays in charge.
        assert_eq!(
            lease::lease_plan(&HostEnvelope::declared_target(HostClass::Other)),
            None
        );
        // The envelope is the DECLARED one, never a live reading: the same
        // host gives the same plan whatever the process holds.
        assert_eq!(
            lease::lease_plan(&host()),
            lease::lease_plan(&host()),
            "static"
        );
    }

    /// The live authority's denial paths on a small and a large declared
    /// host, and conservation: outstanding never passes the envelope.
    #[test]
    fn router_r4_lease_denial_paths_and_conservation() {
        // 48 GiB host (E = 32 GiB): terminal 16, pool 16 -> ONE heavy lane.
        let small = lease::lease_plan(&HostEnvelope {
            class: HostClass::Other,
            threads: 8,
            mem_bytes: 48 * GIB,
            reserve_bytes: 16 * GIB,
            device: DeviceQual::None,
            declared: false,
        })
        .expect("known");
        let a = LeaseAuthority::new();
        assert!(a.arm(small).expect("arm").admits());
        assert!(a
            .acquire(LeaseHolder::MarginRowConcurrent)
            .expect("armed")
            .admits());
        // The concurrent worker holds the pool: the tail and the VJP wave are
        // refused, and a refusal reserves nothing.
        for holder in [LeaseHolder::PostVerifierTail, LeaseHolder::OrtRefineWave] {
            assert!(!a.acquire(holder).expect("armed").admits(), "{holder:?}");
            assert_eq!(a.held(holder), 0);
        }
        // The verifier returns: the tail now fits beside the worker.
        a.release(LeaseHolder::Terminal);
        assert!(a
            .acquire(LeaseHolder::PostVerifierTail)
            .expect("armed")
            .admits());
        // Randomized conservation on the m5 envelope.
        let big = lease::lease_plan(&HostEnvelope::declared_target(HostClass::M5)).expect("m5");
        let mut rng = Rng(0x5eed_4401);
        for _ in 0..200 {
            let a = LeaseAuthority::new();
            a.arm(big).expect("arm");
            for _ in 0..40 {
                let holder = LeaseHolder::ALL[rng.below(lease::HOLDERS as u64) as usize];
                if rng.coin() {
                    let _ = a.acquire(holder);
                } else {
                    let _ = a.release(holder);
                }
                let snap = a.snapshot();
                assert!(snap.outstanding_bytes <= big.envelope_bytes);
                assert!(snap.peak_bytes <= big.envelope_bytes);
                let sum: u64 = snap.holders.iter().map(|h| h.held_bytes).sum();
                assert_eq!(sum, snap.outstanding_bytes);
            }
        }
    }

    /// The lease is admission control only. Its commitments carry bytes and
    /// host facts, never a value a lane computes with; the plan and the R2
    /// commitments are identical whatever the lease does; and with the router
    /// not acting nothing is leased at all.
    #[test]
    fn router_r4_lease_commitments_carry_bytes_and_host_facts_only() {
        let input = row(100_000, Vec::new());
        let (p_on, r2_on, on) = commits(
            &input,
            EffectiveMode::Acting,
            &RuleSwitches::ALL_ON,
            &R4Switches::ALL_ON,
        );
        let (p_off, r2_off, off) = commits(
            &input,
            EffectiveMode::Acting,
            &RuleSwitches::ALL_ON,
            &R4Switches {
                lease: false,
                far_hand_on: false,
            },
        );
        assert!(on.lease.is_some() && off.lease.is_none());
        assert_eq!(p_on, p_off, "the plan never reads the lease");
        assert_eq!(r2_on, r2_off, "R2's commitments never read the lease");
        assert!(p_on.emitted.is_empty(), "no knob value is emitted");
        // The serialized lease plan names only bytes and host facts.
        let text = serde_json::to_string(&on.lease).expect("serialize");
        for field in [
            "envelope_bytes",
            "host",
            "threads",
            "device",
            "declared",
            "needs",
        ] {
            assert!(text.contains(field), "{field} in {text}");
        }
        let value: serde_json::Value = serde_json::from_str(&text).expect("json");
        assert_eq!(
            value.as_object().map(serde_json::Map::len),
            Some(6),
            "the lease plan carries exactly those six fields: {text}"
        );
        for mode in [EffectiveMode::Off, EffectiveMode::Shadow] {
            let (_, _, c) = commits(&input, mode, &RuleSwitches::ALL_ON, &R4Switches::ALL_ON);
            assert!(
                c.lease.is_none() && c.far.is_none(),
                "{mode:?} leases nothing"
            );
        }
    }

    // ------------------------------------------------------------------
    // Acting set 3's commitments.
    // ------------------------------------------------------------------

    #[test]
    fn router_r4_commitments_are_pure_and_deterministic() {
        let mut rng = Rng(0x5eed_4402);
        for _ in 0..400 {
            let input = random_input(&mut rng);
            let p = plan::plan(&input);
            let r2 = acting::commit(&p, &input, EffectiveMode::Acting, &RuleSwitches::ALL_ON);
            for mode in MODES {
                let switches = R4Switches {
                    lease: rng.coin(),
                    far_hand_on: rng.coin(),
                };
                let a = r4::commit(&p, &input, mode, &switches, &r2);
                let b = r4::commit(&p, &input, mode, &switches, &r2);
                assert_eq!(a, b, "commit is a pure function of its input");
                assert_eq!(a.sha.len(), 64);
                assert_eq!(a.rules.len(), R4_RULES.len(), "every rule decides");
                for (act, id) in a.rules.iter().zip(R4_RULES) {
                    assert_eq!(act.rule, id.as_str());
                    assert_eq!(act.acts, act.blocked_by.is_none(), "{act:?}");
                }
                // CR-F acts exactly when it has a route.
                assert_eq!(
                    a.acts(plan::CR_F),
                    a.far.as_ref().is_some_and(|f| !f.routes.is_empty())
                );
                assert_eq!(a.acts(plan::CR_L), a.lease.is_some());
            }
        }
    }

    #[test]
    fn router_r4_nothing_acts_unless_the_mode_is_acting() {
        let input = row(100_000, Vec::new());
        for mode in [EffectiveMode::Off, EffectiveMode::Shadow] {
            let (_, _, c) = commits(&input, mode, &RuleSwitches::ALL_ON, &R4Switches::ALL_ON);
            assert!(!c.acting && c.lease.is_none() && c.far.is_none());
            for act in &c.rules {
                assert!(!act.acts);
                assert_eq!(
                    act.blocked_by.as_deref(),
                    Some(format!("mode:{}", mode.token()).as_str())
                );
            }
        }
        let (_, _, on) = commits(
            &input,
            EffectiveMode::Acting,
            &RuleSwitches::ALL_ON,
            &R4Switches::ALL_ON,
        );
        assert!(
            on.acting && on.acts(plan::CR_L),
            "positive control: CR-L acts"
        );
        // No registry entry is Sealed: CR-F arms in observe mode, does not act.
        let far = rule(&on, "CR-F");
        assert!(!far.acts);
        assert_eq!(far.blocked_by.as_deref(), Some("no_sealed_recipient"));
        assert_eq!(
            on.far,
            Some(FarCommit {
                routes: Vec::new(),
                observe_only: true
            })
        );
    }

    #[test]
    fn router_r4_each_rule_key_ablates_its_own_rule() {
        let input = row(100_000, Vec::new());
        let all = commits(
            &input,
            EffectiveMode::Acting,
            &RuleSwitches::ALL_ON,
            &R4Switches::ALL_ON,
        )
        .2;
        for (id, switches) in [
            (
                "CR-L",
                R4Switches {
                    lease: false,
                    ..R4Switches::ALL_ON
                },
            ),
            (
                "CR-F",
                R4Switches {
                    far_hand_on: false,
                    ..R4Switches::ALL_ON
                },
            ),
        ] {
            let c = commits(
                &input,
                EffectiveMode::Acting,
                &RuleSwitches::ALL_ON,
                &switches,
            )
            .2;
            let own = rule(&c, id);
            assert!(!own.acts, "{id} off must not act");
            assert_eq!(own.blocked_by.as_deref(), Some("key"), "{id}");
            assert_ne!(c.sha, all.sha, "{id} off must change the commitments");
            let other = if id == "CR-L" { "CR-F" } else { "CR-L" };
            assert_eq!(rule(&c, other), rule(&all, other), "{id} moved {other}");
        }
        assert!(R4Switches::ALL_ON.get(plan::CR_L) && R4Switches::ALL_ON.get(plan::CR_F));
        assert!(!R4Switches::ALL_ON.get(plan::CR_K), "not an R4 rule");
    }

    #[test]
    fn router_r4_far_rides_on_cr1_and_never_on_a_short_row() {
        let input = row(100_000, Vec::new());
        let (_, _, no_cr1) = commits(
            &input,
            EffectiveMode::Acting,
            &RuleSwitches {
                cap_authority: false,
                ..RuleSwitches::ALL_ON
            },
            &R4Switches::ALL_ON,
        );
        assert_eq!(rule(&no_cr1, "CR-F").blocked_by.as_deref(), Some("CR-1"));
        assert!(
            no_cr1.far.is_none(),
            "not even observe mode without the cells"
        );
        assert!(no_cr1.acts(plan::CR_L), "the lease needs no cell");
        let short = row(25_000, Vec::new());
        let (p, _, c) = commits(
            &short,
            EffectiveMode::Acting,
            &RuleSwitches::ALL_ON,
            &R4Switches::ALL_ON,
        );
        assert!(p.short_row);
        assert_eq!(rule(&c, "CR-F").blocked_by.as_deref(), Some("short_row"));
        assert!(c.far.is_none());
        // CR-L's envelope must be known.
        let mut unknown = row(100_000, Vec::new());
        unknown.host = HostEnvelope::declared_target(HostClass::Other);
        let (_, _, c) = commits(
            &unknown,
            EffectiveMode::Acting,
            &RuleSwitches::ALL_ON,
            &R4Switches::ALL_ON,
        );
        assert_eq!(
            rule(&c, "CR-L").blocked_by.as_deref(),
            Some("envelope_unknown")
        );
        assert!(c.lease.is_none());
    }

    /// CR-F fires only toward a STRICTLY sealed recipient (S7 + S8): a
    /// `SealedAsShipped` one is today's route, not evidence it converts
    /// handed-on time. And the route needs two distinct governed cells.
    #[test]
    fn router_r4_far_fires_only_toward_strictly_sealed_recipients() {
        let sealed_sink = |ledger: LedgerState| {
            let mut input = row(100_000, Vec::new());
            // lowdim sits before margin-row in the structural order, as in the
            // registry; margin-row is recast as the candidate recipient with
            // the ledger under test (it owns two cells, so only the strict
            // seal check and then the slot check matter).
            let at = input
                .capabilities
                .iter()
                .position(|c| c.id == "margin-row")
                .expect("margin-row");
            input.capabilities.insert(
                at,
                prover(
                    "lowdim",
                    LedgerState::SealedAsShipped,
                    ProgressKind::Volume,
                    AdmissionState::Unknown("needs-loaded-graph".to_owned()),
                ),
            );
            for c in &mut input.capabilities {
                if c.id == "margin-row" {
                    c.ledger = ledger.clone();
                    c.admission = AdmissionState::Admit;
                }
            }
            input
        };
        for ledger in [
            LedgerState::Open("S7".to_owned()),
            LedgerState::Shadow,
            LedgerState::SealedAsShipped,
        ] {
            let input = sealed_sink(ledger.clone());
            let p = plan::plan(&input);
            assert_eq!(
                checkpoint::far_recipient(&p, "lowdim").map(|c| c.id.clone()),
                Err(FarBlock::NoSealedRecipient),
                "{ledger:?} is not a CR-F recipient"
            );
        }
        let input = sealed_sink(LedgerState::Sealed);
        let p = plan::plan(&input);
        assert_eq!(
            checkpoint::far_recipient(&p, "lowdim").map(|c| c.id.clone()),
            Ok("margin-row".to_owned()),
            "a strictly sealed, admitted prover is a recipient"
        );
        // ...but margin-row owns two cells (concurrent, inline): no route.
        let (_, _, c) = commits(
            &input,
            EffectiveMode::Acting,
            &RuleSwitches::ALL_ON,
            &R4Switches::ALL_ON,
        );
        assert!(c.far.as_ref().is_some_and(|f| f.routes.is_empty()));
        assert_eq!(r4::unique_slot("margin-row"), None);
        assert_eq!(r4::unique_slot("lowdim"), Some(LaneSlot::Verifier));
        assert_eq!(
            r4::unique_slot("attack.upfront_apgd"),
            Some(LaneSlot::AttackUpfront)
        );
        // An attack is never a CR-F source: CR-ATK defers it.
        let mut attack = row(100_000, Vec::new());
        for c in &mut attack.capabilities {
            if c.id == "attack.upfront_apgd" {
                c.progress = ProgressKind::Margin;
            }
        }
        let p = plan::plan(&attack);
        assert_eq!(
            checkpoint::far_recipient(&p, "attack.upfront_apgd").map(|c| c.id.clone()),
            Err(FarBlock::AttackDefers)
        );
        // A source that declined itself never hands on.
        let mut declined = sealed_sink(LedgerState::Sealed);
        for c in &mut declined.capabilities {
            if c.id == "lowdim" {
                c.admission = AdmissionState::Decline("too-many-dims".to_owned());
            }
        }
        let p = plan::plan(&declined);
        assert_eq!(
            checkpoint::far_recipient(&p, "lowdim").map(|c| c.id.clone()),
            Err(FarBlock::Declined)
        );
    }

    // ------------------------------------------------------------------
    // FAR detection and the live hand-on.
    // ------------------------------------------------------------------

    fn volume(cap: &'static str, points: &[(f64, f64, u64)]) -> Vec<ProgressSample> {
        points
            .iter()
            .enumerate()
            .map(|(i, &(t, vol, open))| {
                crate::far_authority::volume_sample(CapId(cap), i as u64, vol, open).at_secs(t)
            })
            .collect()
    }

    /// FAR on synthetic progress streams, per kind, in WORK UNITS: only the
    /// rate decides; a converging stream is never FAR.
    #[test]
    fn router_r4_far_detection_on_synthetic_progress_streams() {
        // Volume: certified volume rising (converging) is not FAR even while
        // the queue grows; flat volume with a growing queue is FAR.
        let converging = volume(
            "v",
            &[
                (0.0, 0.10, 4),
                (1.0, 0.30, 8),
                (2.0, 0.55, 16),
                (3.0, 0.80, 32),
            ],
        );
        let t = project::trailing(&converging, "v", 3.0, 1);
        assert_eq!(
            project::classify(ProgressKind::Volume, &t),
            FarClass::NotFar
        );
        let flat = volume(
            "v",
            &[
                (0.0, 0.40, 4),
                (1.0, 0.4001, 8),
                (2.0, 0.4002, 16),
                (3.0, 0.4003, 32),
            ],
        );
        let t = project::trailing(&flat, "v", 3.0, 1);
        assert_eq!(project::classify(ProgressKind::Volume, &t), FarClass::Far);
        // Flat volume with a SHRINKING queue (the tree is closing) is not FAR.
        let draining = volume(
            "v",
            &[
                (0.0, 0.40, 32),
                (1.0, 0.40, 16),
                (2.0, 0.40, 8),
                (3.0, 0.40, 4),
            ],
        );
        let t = project::trailing(&draining, "v", 3.0, 1);
        assert_eq!(
            project::classify(ProgressKind::Volume, &t),
            FarClass::NotFar
        );
        // FrontierBinary: a collapse (no closes, frontier growing) is FAR; a
        // closing tree (closes per expansion above T_FAR) is not.
        let collapse = collapsing("f", 6.0);
        let t = project::trailing(&collapse, "f", 6.0, project::FRONTIER_WINDOW_UNITS);
        assert_eq!(
            project::classify(ProgressKind::FrontierBinary, &t),
            FarClass::Far
        );
        let closing: Vec<ProgressSample> = (0..=6u64)
            .map(|i| {
                let exp = i * 20;
                ProgressSample::frontier(CapId("f"), exp, 2 * exp + 1, exp, exp / 2 + 1)
                    .at_secs(i as f64)
            })
            .collect();
        let t = project::trailing(&closing, "f", 6.0, project::FRONTIER_WINDOW_UNITS);
        assert_eq!(
            project::classify(ProgressKind::FrontierBinary, &t),
            FarClass::NotFar
        );
        // Bfs: growth >= 1 past depth 8 is FAR; heavy pruning is not.
        let bfs = |pruned_per_level: u64| -> Vec<ProgressSample> {
            let (mut n, mut p) = (1u64, 0u64);
            (9..=12u64)
                .map(|depth| {
                    n += 64;
                    p += pruned_per_level;
                    ProgressSample::bfs(CapId("b"), depth, n, p).at_secs(depth as f64)
                })
                .collect()
        };
        let t = project::trailing(&bfs(8), "b", 12.0, 1);
        assert_eq!(project::classify(ProgressKind::Bfs, &t), FarClass::Far);
        let t = project::trailing(&bfs(60), "b", 12.0, 1);
        assert_eq!(project::classify(ProgressKind::Bfs, &t), FarClass::NotFar);
        // Opaque / yield / dual: never FAR.
        for kind in [
            ProgressKind::Opaque,
            ProgressKind::Yield,
            ProgressKind::Dual,
        ] {
            assert_eq!(project::classify(kind, &t), FarClass::Unknown, "{kind:?}");
        }
    }

    /// A plan whose source is the fixture's FrontierBinary capability (with
    /// safe points) and whose sink is strictly sealed, with c2 placed at
    /// `c2_s` seconds (c3 never reached).
    fn fixture_plan(c2_s: u64) -> Plan {
        let mut input = super::hand_on_input();
        input.capabilities[0].id = FIXTURE_CAP.to_owned();
        let mut p = plan::plan(&input);
        for c in &mut p.checkpoints {
            match c.id {
                CheckpointId::C1 => c.at_ms = 0,
                CheckpointId::C2 => c.at_ms = c2_s * 1_000,
                CheckpointId::C3 => c.at_ms = u64::MAX / 4,
            }
        }
        p
    }

    fn armed_cells(anchor: Instant) -> Authority {
        let cells = Authority::new();
        cells
            .arm(
                Frame::new(
                    anchor,
                    anchor + Duration::from_secs(1_000),
                    [Duration::from_secs(5); SLOTS],
                    Duration::from_secs(2),
                ),
                [true; SLOTS],
                Some(LaneSlot::Verifier),
                None,
            )
            .expect("arm cells");
        cells
            .claim_deadline(LaneSlot::Verifier, anchor + Duration::from_mins(15), anchor)
            .expect("the source's cell is claimed");
        cells
    }

    fn route() -> FarCommit {
        FarCommit {
            routes: vec![FarRoute {
                from: FIXTURE_CAP.to_owned(),
                to: "sink".to_owned(),
                from_slot: LaneSlot::Verifier,
                to_slot: LaneSlot::MarginRowInline,
            }],
            observe_only: false,
        }
    }

    /// Feed one fixture record into a live authority at every sample (the
    /// lane's unit boundaries), with c2 at sample `k`. Returns the first
    /// stopping action and the recorded decision at c2.
    fn live(
        record: &super::Record,
        k: u64,
    ) -> (Option<FarAction>, Option<checkpoint::FarDecision>) {
        let anchor = Instant::now();
        let cells = armed_cells(anchor);
        let far = FarAuthority::new();
        let p = fixture_plan(k);
        far.arm(p, route(), anchor);
        let mut stop = None;
        for (i, sample) in record.samples.iter().enumerate() {
            let action = far.decide_at(
                *sample,
                At::Unit(i as u64),
                sample.secs(),
                &cells,
                anchor + Duration::from_secs(i as u64),
            );
            if action.stops() {
                stop = Some(action);
                break;
            }
        }
        let decision = far.snapshot().events.first().map(|e| e.decision.clone());
        (stop, decision)
    }

    /// The R0 fixtures through the LIVE path: at every possible c2 position,
    /// the live decision equals the retrospective classification (so R1's
    /// 0/8 transfers), no eventually-unsat record is ever handed on
    /// (false-FAR = 0), and the collapsed timeouts are (positive control).
    #[test]
    fn router_r4_live_far_flags_no_unsat_fixture_and_matches_retrospective() {
        let records = fixtures();
        let mut unsat_decisions = 0usize;
        let mut fired = Vec::new();
        for record in &records {
            let retro = super::classifications(record);
            for k in 1..record.samples.len() as u64 {
                let (stop, decision) = live(record, k);
                let decision = decision.expect("c2 is evaluated once the lane passes it");
                assert_eq!(decision.at, CheckpointId::C2);
                assert_eq!(
                    decision.class, retro[k as usize],
                    "{} at {k}: live and retrospective disagree",
                    record.source
                );
                if record.verdict == "unsat" {
                    unsat_decisions += 1;
                    assert_ne!(decision.class, FarClass::Far, "{} at {k}", record.source);
                    assert!(stop.is_none(), "{} (unsat) handed on at {k}", record.source);
                } else if stop.is_some() {
                    fired.push((record.source.clone(), k));
                }
            }
        }
        assert!(unsat_decisions >= 8, "the unsat fixtures were evaluated");
        assert!(
            fired.iter().any(|(s, _)| s.ends_with("MRG_c8330_ctl_r1")),
            "control: c8330 collapses and is handed on: {fired:?}"
        );
        assert!(
            fired.iter().any(|(s, _)| s.ends_with("CCT_1440_w_r1")),
            "control: c1440 never closes and is handed on: {fired:?}"
        );
        assert!(
            !fired.iter().any(|(s, _)| s.ends_with("FDB_c3494_ctl_r1")),
            "c3494's windows never sit at or below T_FAR twice: {fired:?}"
        );
    }

    /// A live hand-on conserves time: the recipient is credited at most the
    /// source's unspent cell (never past its own cap), the source stays
    /// stopped for the row, the cells refuse a started / identical /
    /// ungoverned recipient, and the decision replays from its samples.
    #[test]
    fn router_r4_far_hand_on_conserves_time_and_replays() {
        let collapse = fixtures()
            .into_iter()
            .find(|r| r.source.ends_with("CCT_1440_w_r1"))
            .expect("fixture");
        let anchor = Instant::now();
        let cells = armed_cells(anchor);
        let far = FarAuthority::new();
        let k = super::classifications(&collapse)
            .iter()
            .position(|c| *c == FarClass::Far)
            .expect("the collapse reads FAR somewhere") as u64;
        far.arm(fixture_plan(k), route(), anchor);
        let mut stopped = None;
        for (i, sample) in collapse.samples.iter().enumerate() {
            let action = far.decide_at(
                *sample,
                At::Unit(i as u64),
                sample.secs(),
                &cells,
                anchor + Duration::from_secs(i as u64),
            );
            if let FarAction::HandOn {
                to,
                unspent_ms,
                credited_ms,
            } = action
            {
                stopped = Some((to, unspent_ms, credited_ms));
                break;
            }
        }
        let (to, unspent_ms, credited_ms) = stopped.expect("the collapse hands on");
        assert_eq!(to, "sink");
        assert!(credited_ms <= unspent_ms, "{credited_ms} <= {unspent_ms}");
        assert!(credited_ms > 0);
        let snap = cells.snapshot();
        let hand = snap.hand_ons.iter().find(|h| h.far).expect("ledgered");
        assert_eq!(hand.from, LaneSlot::Verifier);
        assert_eq!(hand.to, Some(LaneSlot::MarginRowInline));
        assert!(hand.credited_ns <= hand.unspent_ns);
        let inline = snap
            .slots
            .iter()
            .find(|s| s.slot == LaneSlot::MarginRowInline)
            .expect("slot");
        assert_eq!(inline.credit_ms, credited_ms);
        assert!(inline.credit_ms <= inline.cap_ms, "never past its cap");
        // The source stays stopped for the row, and nothing is handed twice.
        let again = far.decide_at(
            *collapse.samples.last().expect("sample"),
            At::Unit(999),
            1e6,
            &cells,
            anchor + Duration::from_secs(10),
        );
        assert!(again.stops());
        assert_eq!(
            cells.snapshot().hand_ons.iter().filter(|h| h.far).count(),
            1
        );
        // Replay: the recorded decision re-derives from its recorded samples.
        let event = far
            .snapshot()
            .events
            .into_iter()
            .find(|e| e.handed)
            .expect("event");
        let samples: Vec<ProgressSample> = event
            .samples
            .iter()
            .map(|s| s.to_sample(Some(CapId(FIXTURE_CAP))).expect("sample"))
            .collect();
        assert_eq!(
            checkpoint::far_decision(&fixture_plan(k), &samples, FIXTURE_CAP, event.decision.at),
            event.decision
        );
        // The cells refuse a recipient that started, the same slot, or an
        // ungoverned one — and then release nothing.
        let cells = armed_cells(anchor);
        let now = anchor + Duration::from_secs(5);
        assert!(cells
            .release_to(LaneSlot::Verifier, LaneSlot::Verifier, now)
            .is_none());
        cells
            .claim_deadline(
                LaneSlot::MarginRowInline,
                anchor + Duration::from_secs(990),
                anchor,
            )
            .expect("recipient started");
        assert!(cells
            .release_to(LaneSlot::Verifier, LaneSlot::MarginRowInline, now)
            .is_none());
        assert!(cells.snapshot().hand_ons.is_empty(), "nothing was released");
        // Observe mode (no route): classified and recorded, never handed on.
        let far = FarAuthority::new();
        let cells = armed_cells(anchor);
        far.arm(
            fixture_plan(k),
            FarCommit {
                routes: Vec::new(),
                observe_only: true,
            },
            anchor,
        );
        for (i, sample) in collapse.samples.iter().enumerate() {
            assert!(!far
                .decide_at(
                    *sample,
                    At::Unit(i as u64),
                    sample.secs(),
                    &cells,
                    anchor + Duration::from_secs(i as u64)
                )
                .stops());
        }
        let snap = far.snapshot();
        assert_eq!(snap.handed, 0);
        assert!(snap
            .events
            .iter()
            .any(|e| e.decision.class == FarClass::Far && !e.handed));
        assert_eq!(cells.snapshot().hand_ons.len(), 0);
    }

    /// §6.3: a contained fault disarms the lease and FAR authorities, so every
    /// gate keeps today's live decision and no lane is handed on.
    #[test]
    fn router_r4_fault_disarms_the_lease_and_far_authorities() {
        let _serial = globals();
        guard::reset();
        let lease_authority = crate::memory_lease::global();
        let far = crate::far_authority::global();
        lease_authority
            .arm(lease::lease_plan(&host()).expect("known"))
            .expect("arm lease");
        far.arm(
            fixture_plan(1),
            FarCommit {
                routes: Vec::new(),
                observe_only: true,
            },
            Instant::now(),
        );
        assert!(
            lease_authority.is_armed() && far.is_armed(),
            "positive control"
        );
        let fault = guard::fail_open("injected-r4", || -> u32 {
            panic!("router-r4 injected panic")
        })
        .expect_err("the injected panic is contained");
        assert!(!fault.skipped);
        assert!(!lease_authority.is_armed(), "the lease is disarmed");
        assert!(!far.is_armed(), "the FAR authority is disarmed");
        assert_eq!(lease_authority.acquire(LeaseHolder::PostVerifierTail), None);
        assert_eq!(
            crate::far_authority::at_safe_point(
                ProgressSample::frontier(CapId(FIXTURE_CAP), 0, 1, 0, 1),
                0
            ),
            FarAction::Continue
        );
        guard::reset();
        super::super::bus::reset();
    }

    #[test]
    fn router_r4_arithmetic_never_panics_on_extremes() {
        let mut rng = Rng(0x5eed_4403);
        for _ in 0..300 {
            let mut input = random_input(&mut rng);
            input.host.mem_bytes = [0, 1, u64::MAX, rng.next()][rng.below(4) as usize];
            input.host.reserve_bytes = [0, u64::MAX, rng.next()][rng.below(3) as usize];
            let p = plan::plan(&input);
            let r2 = acting::commit(&p, &input, EffectiveMode::Acting, &RuleSwitches::ALL_ON);
            let _ = r4::commit(&p, &input, EffectiveMode::Acting, &R4Switches::ALL_ON, &r2);
            let _ = r4::far_routes(&p, &r2);
            let _ = lease::lease_plan(&input.host);
            let _ = decide(
                rng.next(),
                LeaseNeed {
                    want_bytes: rng.next(),
                    min_bytes: rng.next(),
                },
            );
        }
        let far = FarAuthority::new();
        let cells = Authority::new();
        far.arm(fixture_plan(0), route(), Instant::now());
        for t in [0.0, -1.0, f64::NAN, f64::INFINITY, 1e300] {
            let _ = far.decide_at(
                ProgressSample::frontier(CapId(FIXTURE_CAP), u64::MAX, u64::MAX, u64::MAX, 0),
                At::Unit(u64::MAX),
                t,
                &cells,
                Instant::now(),
            );
        }
        let _ = far.snapshot();
    }
}

/// #router-r5 (ROUTER_PROGRAM §3.5, §10 R5): acting set 4 — CR-RS (capabilities
/// routed, not armed: only at SR.seal) and CR-G (gap-order contracts).
mod r5 {
    use super::super::decl::{
        Admission, AdmitWhen, DeviceUse, KnobRisk, OpEnvelope, OpKind, OpSet, SpecArchetype,
        SpecSet, SrLadder, SrLevel, DECLINE_DEVICE, DECLINE_NO_OPERATOR, DECLINE_PARAMS,
        DECLINE_P_EFF, DECLINE_SPEC, DECLINE_THREADS, UNKNOWN_FACTS, UNKNOWN_OPS, UNKNOWN_SIZE,
    };
    use super::super::facts::{DeviceQual, InstanceFacts};
    use super::super::gap_contract::{
        class_window, rung_ns, ClassCost, ClassWindow, CONTRACT_LEVELS,
    };
    use super::super::plan::{
        AdmissionState, BudgetView, CapClass, CapInput, LedgerState, PolicyInput, RouteSpec,
    };
    use super::super::route::{
        self, may_auto_select, sealed, R5Input, R5Switches, R5_RULES, R5_SCHEMA,
    };
    use super::super::{EffectiveMode, PRIORS_VERSION, ROUTER_SCHEMA};
    use super::{cap, globals, guard, host, random_input, Rng};
    use crate::route_authority::RouteAuthority;

    const ROBUSTNESS: SpecSet = SpecSet::EMPTY.with(SpecArchetype::Robustness);

    fn conv_relu() -> OpSet {
        OpSet::EMPTY
            .with(OpKind::Conv)
            .with(OpKind::Relu)
            .with(OpKind::Linear)
    }

    /// Known facts with a COMPLETE signature `ops` and an unknown archetype.
    fn facts(ops: OpSet) -> InstanceFacts {
        let mut facts = InstanceFacts::known_empty();
        facts.ops = ops;
        facts.ops_complete = true;
        facts.params = Some(3_000_000);
        facts.p_eff = Some(3072);
        facts
    }

    struct Spec {
        sr: SrLevel,
        risk: KnobRisk,
        admit: AdmitWhen,
        envelope_ops: OpSet,
        envelope_specs: SpecSet,
        key: &'static str,
        slot: Option<&'static str>,
        parent: Option<&'static str>,
    }

    /// A sealed, Low-risk, robustness-serving capability covering conv/relu.
    fn sealed_spec(key: &'static str) -> Spec {
        Spec {
            sr: SrLevel::Seal,
            risk: KnobRisk::Low,
            admit: AdmitWhen {
                specs: ROBUSTNESS,
                ..AdmitWhen::ANY
            },
            envelope_ops: conv_relu(),
            envelope_specs: ROBUSTNESS,
            key,
            slot: None,
            parent: Some("lane"),
        }
    }

    fn routable(id: &str, ledger: LedgerState, spec: &Spec) -> CapInput {
        CapInput {
            route: Some(RouteSpec {
                sr: spec.sr,
                parent: spec.parent.map(str::to_owned),
                admit_when: spec.admit,
                envelope_ops: spec.envelope_ops,
                envelope_specs: spec.envelope_specs,
                risk: spec.risk,
                settings: vec![(spec.key.to_owned(), "true".to_owned())],
                slot: spec.slot.map(str::to_owned),
            }),
            ..cap(
                id,
                CapClass::Tier {
                    parent: "lane".to_owned(),
                },
                ledger,
                super::super::decl::ProgressKind::Opaque,
                None,
                AdmissionState::Admit,
            )
        }
    }

    /// The parent lane, admitted inline (re-checked at its own site).
    fn lane(admission: AdmissionState) -> CapInput {
        cap(
            "lane",
            CapClass::Lane(super::super::decl::LaneRole::Prover { terminal_ok: true }),
            LedgerState::SealedAsShipped,
            super::super::decl::ProgressKind::FrontierBinary,
            None,
            admission,
        )
    }

    fn input(ops: OpSet, capabilities: Vec<CapInput>) -> PolicyInput {
        PolicyInput {
            schema: ROUTER_SCHEMA.to_owned(),
            facts: facts(ops),
            host: host(),
            budget: BudgetView {
                official_ms: 100_000,
                internal_ms: 95_000,
                join_margin_ms: 2_000,
            },
            capabilities,
            pins: Vec::new(),
            priors_version: PRIORS_VERSION.to_owned(),
            registry_version: "0123abcd".to_owned(),
            priors: None,
        }
    }

    fn inline() -> AdmissionState {
        AdmissionState::Unknown(super::super::decl::UNKNOWN_INLINE.to_owned())
    }

    fn acting(input: &PolicyInput, pins: &[&str]) -> route::R5Commitments {
        route::commit(
            input,
            &R5Input {
                pinned_keys: pins.iter().map(|p| (*p).to_owned()).collect(),
            },
            EffectiveMode::Acting,
            &R5Switches::ALL_ON,
        )
    }

    /// The routing precondition on synthetic facts: every field refuses on
    /// its own, unknown facts are refusals (never guesses), and an unknown
    /// specification archetype is DEFERRED to the consumer's call site.
    #[test]
    fn router_r5_admission_predicates_on_synthetic_facts() {
        let h = host();
        let any = AdmitWhen::ANY;
        assert_eq!(
            any.evaluate(&InstanceFacts::unknown("scan>256MiB"), &h),
            Admission::Unknown(UNKNOWN_FACTS)
        );
        let mut lower_bound = facts(conv_relu());
        lower_bound.ops_complete = false;
        assert_eq!(
            any.evaluate(&lower_bound, &h),
            Admission::Unknown(UNKNOWN_OPS)
        );
        assert_eq!(any.evaluate(&facts(conv_relu()), &h), Admission::Admit);

        let needs_conv = AdmitWhen {
            ops_any: OpSet::EMPTY.with(OpKind::Conv),
            ..any
        };
        assert_eq!(
            needs_conv.evaluate(&facts(OpSet::EMPTY.with(OpKind::Linear)), &h),
            Admission::Decline(DECLINE_NO_OPERATOR)
        );
        assert_eq!(
            needs_conv.evaluate(&facts(conv_relu()), &h),
            Admission::Admit
        );

        let robust = AdmitWhen {
            specs: ROBUSTNESS,
            ..any
        };
        let mut halfspace = facts(conv_relu());
        halfspace.spec = Some(SpecArchetype::Halfspace);
        assert_eq!(
            robust.evaluate(&halfspace, &h),
            Admission::Decline(DECLINE_SPEC)
        );
        halfspace.spec = Some(SpecArchetype::Robustness);
        assert_eq!(robust.evaluate(&halfspace, &h), Admission::Admit);
        assert_eq!(
            robust.evaluate(&facts(conv_relu()), &h),
            Admission::Admit,
            "an unknown archetype is deferred to the consumer's site, not refused"
        );

        let small = AdmitWhen {
            max_params: Some(1_000_000),
            max_p_eff: Some(16),
            ..any
        };
        assert_eq!(
            small.evaluate(&facts(conv_relu()), &h),
            Admission::Decline(DECLINE_PARAMS)
        );
        let mut tiny = facts(conv_relu());
        tiny.params = Some(10);
        assert_eq!(small.evaluate(&tiny, &h), Admission::Decline(DECLINE_P_EFF));
        tiny.p_eff = Some(5);
        assert_eq!(small.evaluate(&tiny, &h), Admission::Admit);
        tiny.params = None;
        assert_eq!(
            small.evaluate(&tiny, &h),
            Admission::Unknown(UNKNOWN_SIZE),
            "a ceiling over an unmeasured size is a refusal"
        );

        let device = AdmitWhen {
            device: DeviceUse::Search,
            min_threads: 8,
            ..any
        };
        let mut cpu = h;
        cpu.device = DeviceQual::None;
        assert_eq!(
            device.evaluate(&facts(conv_relu()), &cpu),
            Admission::Decline(DECLINE_DEVICE)
        );
        let mut narrow = h;
        narrow.threads = 4;
        assert_eq!(
            device.evaluate(&facts(conv_relu()), &narrow),
            Admission::Decline(DECLINE_THREADS)
        );
        assert_eq!(device.evaluate(&facts(conv_relu()), &h), Admission::Admit);
    }

    /// SR is DERIVED: `seal` needs S7 AND S8 AND a recorded envelope.
    #[test]
    fn router_r5_sr_level_is_derived_from_the_ladder() {
        let envelope = OpEnvelope {
            ops: conv_relu(),
            specs: ROBUSTNESS,
        };
        let ladder = |s7, s8, obs| SrLadder { s7, s8, obs };
        assert_eq!(SrLadder::DECLARED.level(envelope), SrLevel::Decl);
        assert_eq!(ladder(true, false, false).level(envelope), SrLevel::Decl);
        assert_eq!(ladder(false, true, false).level(envelope), SrLevel::Decl);
        assert_eq!(ladder(false, false, true).level(envelope), SrLevel::Obs);
        assert_eq!(ladder(true, true, false).level(envelope), SrLevel::Seal);
        assert_eq!(
            ladder(true, true, true).level(OpEnvelope::NONE),
            SrLevel::Obs,
            "S7 + S8 without a recorded envelope is not a seal"
        );
        assert_eq!(
            ladder(true, true, false).level(OpEnvelope {
                ops: conv_relu(),
                specs: SpecSet::EMPTY,
            }),
            SrLevel::Decl,
            "an envelope without archetypes is not recorded"
        );
    }

    /// The seal gate. A sealed, admitted, covered, unpinned capability routes;
    /// everything else records EVERY failing check. The two seal checks and
    /// the moat guard are independent: a Low-risk capability at `obs` is
    /// refused by the seal check alone (its moat guard admits it), and a
    /// High-risk capability below seal carries the moat refusal whatever the
    /// seal checks say.
    #[test]
    fn router_r5_routes_only_at_seal_and_records_every_reason() {
        let mut obs_low = sealed_spec("k.obs_low");
        obs_low.sr = SrLevel::Obs;
        let mut decl_high = sealed_spec("k.decl_high");
        decl_high.sr = SrLevel::Decl;
        decl_high.risk = KnobRisk::High;
        let mut uncovered = sealed_spec("k.uncovered");
        uncovered.envelope_ops = OpSet::EMPTY.with(OpKind::Linear);
        let row = input(
            conv_relu(),
            vec![
                lane(inline()),
                routable("ok", LedgerState::Sealed, &sealed_spec("k.ok")),
                routable("obs_low", LedgerState::Sealed, &obs_low),
                routable("decl_high", LedgerState::Open("S7".to_owned()), &decl_high),
                routable("ledger_open", LedgerState::Shadow, &sealed_spec("k.ledger")),
                routable("uncovered", LedgerState::Sealed, &uncovered),
            ],
        );
        let c = acting(&row, &[]);
        assert_eq!(c.schema, R5_SCHEMA);
        let reasons = |cap: &str| -> Vec<String> {
            c.routes
                .iter()
                .find(|d| d.cap == cap)
                .unwrap_or_else(|| panic!("{cap} decided"))
                .reasons
                .clone()
        };
        let routed: Vec<&str> = c.routed().map(|d| d.cap.as_str()).collect();
        assert_eq!(
            routed,
            ["ok"],
            "only the sealed, admitted, covered capability"
        );
        assert!(c.acts(R5_RULES[0]));
        assert_eq!(
            reasons("obs_low"),
            ["sr:obs"],
            "the seal check alone refuses it"
        );
        assert!(may_auto_select(KnobRisk::Low, false));
        let high = reasons("decl_high");
        assert_eq!(high[0], "sr:decl");
        assert!(
            high.contains(&"moat:high_unsealed".to_owned()),
            "the moat guard fires independently: {high:?}"
        );
        assert!(!may_auto_select(KnobRisk::High, false));
        assert!(may_auto_select(KnobRisk::High, true));
        assert_eq!(
            reasons("ledger_open"),
            ["ledger:shadow"],
            "the level alone is not a seal: the ledger must be Sealed too"
        );
        assert_eq!(reasons("uncovered"), ["envelope:ops"]);
        // Routed settings are the declaration's.
        assert_eq!(
            c.routed().next().map(|d| d.settings.clone()),
            Some(vec![("k.ok".to_owned(), "true".to_owned())])
        );
        // `sealed` needs both the level and the ledger.
        let spec = row.capabilities[1].route.clone().expect("spec");
        assert!(sealed(&spec, &LedgerState::Sealed));
        assert!(!sealed(&spec, &LedgerState::SealedAsShipped));
    }

    /// The parent's admission: admitted, or inline (re-checked at its own
    /// site) lets the consumer route; declined or any other unknown refuses.
    #[test]
    fn router_r5_parent_admission_gates_the_route() {
        for (parent, expect) in [
            (AdmissionState::Admit, None),
            (inline(), None),
            (
                AdmissionState::Decline("single-network-property".to_owned()),
                Some("parent:lane:decline:single-network-property"),
            ),
            (
                AdmissionState::Unknown("facts-unknown".to_owned()),
                Some("parent:lane:unknown:facts-unknown"),
            ),
        ] {
            let row = input(
                conv_relu(),
                vec![
                    lane(parent),
                    routable("cap", LedgerState::Sealed, &sealed_spec("k")),
                ],
            );
            let c = acting(&row, &[]);
            let d = &c.routes[0];
            assert_eq!(d.blocked_by(), expect, "{d:?}");
            assert_eq!(d.routed, expect.is_none());
        }
        // An unregistered parent refuses.
        let row = input(
            conv_relu(),
            vec![routable("cap", LedgerState::Sealed, &sealed_spec("k"))],
        );
        assert_eq!(
            acting(&row, &[]).routes[0].blocked_by(),
            Some("parent:lane:unregistered")
        );
    }

    /// Envelope coverage on the archetype: a known archetype must be inside
    /// the envelope; an unknown one needs every archetype the call site
    /// serves inside it (an empty "serves" means all of them).
    #[test]
    fn router_r5_unknown_spec_needs_the_served_archetypes_inside_the_envelope() {
        let mut halfspace_only = sealed_spec("k");
        halfspace_only.envelope_specs = SpecSet::EMPTY.with(SpecArchetype::Halfspace);
        let row = input(
            conv_relu(),
            vec![
                lane(inline()),
                routable("cap", LedgerState::Sealed, &halfspace_only),
            ],
        );
        assert_eq!(
            acting(&row, &[]).routes[0].blocked_by(),
            Some("envelope:spec")
        );
        let mut serves_all = sealed_spec("k");
        serves_all.admit.specs = SpecSet::EMPTY;
        let row = input(
            conv_relu(),
            vec![
                lane(inline()),
                routable("cap", LedgerState::Sealed, &serves_all),
            ],
        );
        assert_eq!(
            acting(&row, &[]).routes[0].blocked_by(),
            Some("envelope:spec"),
            "serving every archetype needs an envelope over every archetype"
        );
        serves_all.envelope_specs = SpecSet::ALL;
        let row = input(
            conv_relu(),
            vec![
                lane(inline()),
                routable("cap", LedgerState::Sealed, &serves_all),
            ],
        );
        assert!(acting(&row, &[]).routes[0].routed);
        // A known archetype outside the envelope refuses.
        let mut known = input(
            conv_relu(),
            vec![
                lane(inline()),
                routable("cap", LedgerState::Sealed, &sealed_spec("k")),
            ],
        );
        known.facts.spec = Some(SpecArchetype::Disjunctive);
        let d = &acting(&known, &[]).routes[0];
        assert!(!d.routed);
        assert!(d
            .reasons
            .contains(&"admit:decline:spec-not-served".to_owned()));
        assert!(d.reasons.contains(&"envelope:spec".to_owned()));
    }

    /// A pin is never overridden; a pin on a SEALED capability is pin debt,
    /// a pin on an unsealed one is not.
    #[test]
    fn router_r5_pins_block_and_a_pinned_sealed_capability_is_pin_debt() {
        let mut unsealed = sealed_spec("k.unsealed");
        unsealed.sr = SrLevel::Decl;
        let row = input(
            conv_relu(),
            vec![
                lane(inline()),
                routable("sealed", LedgerState::Sealed, &sealed_spec("k.sealed")),
                routable("unsealed", LedgerState::Open("S7".to_owned()), &unsealed),
            ],
        );
        let c = acting(&row, &["k.sealed", "k.unsealed"]);
        let sealed_d = &c.routes[0];
        assert!(!sealed_d.routed);
        assert_eq!(sealed_d.reasons, ["pin:k.sealed"]);
        assert!(
            sealed_d.pin_debt,
            "a file arming a sealed capability is pin debt"
        );
        let unsealed_d = &c.routes[1];
        assert!(
            !unsealed_d.pin_debt,
            "an unsealed capability's pin is not debt"
        );
        assert!(unsealed_d.reasons.contains(&"pin:k.unsealed".to_owned()));
        assert_eq!(c.pin_debt().count(), 1);
        // Unpinned, the sealed one routes.
        assert!(acting(&row, &[]).routes[0].routed);
    }

    /// Two variants of one exclusive slot never route together: the first in
    /// registry order takes the slot.
    #[test]
    fn router_r5_exclusive_slot_routes_one_variant() {
        let mut first = sealed_spec("slot.key");
        first.slot = Some("tree_mode");
        let mut second = sealed_spec("slot.key");
        second.slot = Some("tree_mode");
        let row = input(
            conv_relu(),
            vec![
                lane(inline()),
                routable("first", LedgerState::Sealed, &first),
                routable("second", LedgerState::Sealed, &second),
            ],
        );
        let c = acting(&row, &[]);
        assert!(c.routes[0].routed);
        assert_eq!(c.routes[1].reasons, ["slot:tree_mode:first"]);
    }

    #[test]
    fn router_r5_nothing_acts_unless_the_mode_is_acting() {
        let row = input(
            conv_relu(),
            vec![
                lane(inline()),
                routable("ok", LedgerState::Sealed, &sealed_spec("k")),
            ],
        );
        for mode in [EffectiveMode::Off, EffectiveMode::Shadow] {
            let c = route::commit(&row, &R5Input::default(), mode, &R5Switches::ALL_ON);
            assert!(!c.acting);
            assert!(c.routes.is_empty(), "{mode:?} decides no route");
            assert!(!c.gap_contract);
            for rule in R5_RULES {
                assert!(!c.acts(rule));
            }
            assert!(c.rules.iter().all(
                |a| a.blocked_by.as_deref() == Some(format!("mode:{}", mode.token()).as_str())
            ));
        }
        let on = acting(&row, &[]);
        assert!(on.acting && on.gap_contract && on.routed().count() == 1);
    }

    #[test]
    fn router_r5_each_rule_key_ablates_its_own_rule() {
        let row = input(
            conv_relu(),
            vec![
                lane(inline()),
                routable("ok", LedgerState::Sealed, &sealed_spec("k")),
            ],
        );
        let no_route = route::commit(
            &row,
            &R5Input::default(),
            EffectiveMode::Acting,
            &R5Switches {
                route: false,
                gap_contract: true,
            },
        );
        assert!(!no_route.acts(R5_RULES[0]));
        assert_eq!(no_route.rules[0].blocked_by.as_deref(), Some("key"));
        assert_eq!(no_route.routed().count(), 0);
        assert_eq!(no_route.routes[0].reasons, ["key"]);
        assert!(no_route.gap_contract && no_route.acts(R5_RULES[1]));
        let no_gap = route::commit(
            &row,
            &R5Input::default(),
            EffectiveMode::Acting,
            &R5Switches {
                route: true,
                gap_contract: false,
            },
        );
        assert!(!no_gap.gap_contract);
        assert_eq!(no_gap.rules[1].blocked_by.as_deref(), Some("key"));
        assert_eq!(no_gap.routed().count(), 1);
        // With nothing sealed, CR-RS says so.
        let mut unsealed = sealed_spec("k");
        unsealed.sr = SrLevel::Decl;
        let none = acting(
            &input(
                conv_relu(),
                vec![
                    lane(inline()),
                    routable("decl", LedgerState::Open("S7".to_owned()), &unsealed),
                ],
            ),
            &[],
        );
        assert_eq!(
            none.rules[0].blocked_by.as_deref(),
            Some("no_sealed_capability")
        );
    }

    #[test]
    fn router_r5_commitments_are_pure_and_deterministic() {
        let mut rng = Rng(0x5eed_5005);
        for _ in 0..500 {
            let mut row = random_input(&mut rng);
            row.facts.ops_complete = rng.coin();
            for cap in &mut row.capabilities {
                if rng.coin() {
                    let mut spec = sealed_spec("k.rand");
                    spec.sr = [SrLevel::Decl, SrLevel::Obs, SrLevel::Seal][rng.below(3) as usize];
                    spec.risk = if rng.coin() {
                        KnobRisk::High
                    } else {
                        KnobRisk::Low
                    };
                    spec.envelope_ops = OpSet(rng.next());
                    spec.envelope_specs = SpecSet(rng.next() as u32);
                    spec.slot = rng.coin().then_some("slot");
                    spec.parent = None;
                    cap.route = routable(&cap.id, cap.ledger.clone(), &spec).route;
                }
            }
            let pins = R5Input {
                pinned_keys: if rng.coin() {
                    vec!["k.rand".to_owned()]
                } else {
                    Vec::new()
                },
            };
            let a = route::commit(&row, &pins, EffectiveMode::Acting, &R5Switches::ALL_ON);
            let b = route::commit(&row, &pins, EffectiveMode::Acting, &R5Switches::ALL_ON);
            assert_eq!(a, b);
            assert_eq!(a.sha.len(), 64);
            for d in &a.routes {
                assert_eq!(d.routed, d.reasons.is_empty());
                if d.routed {
                    let cap = row
                        .capabilities
                        .iter()
                        .find(|c| c.id == d.cap)
                        .expect("cap");
                    let spec = cap.route.as_ref().expect("route");
                    assert!(
                        sealed(spec, &cap.ledger),
                        "a routed capability is always sealed: {d:?}"
                    );
                    assert!(may_auto_select(spec.risk, true));
                    assert!(row.facts.ops_complete);
                }
            }
        }
    }

    /// CR-G's pure arithmetic: today's share without a price; the share when
    /// it covers the rung; the full rung when it does not and the tier holds
    /// it; a decline when the tier does not.
    #[test]
    fn router_r5_gap_contract_arithmetic() {
        let none = ClassCost::default();
        assert_eq!(rung_ns(&none), None);
        assert_eq!(
            class_window(10, 100, &none),
            ClassWindow::Fair { window_ns: 10 }
        );
        let built = none.with_run(40, 30, 3);
        assert_eq!(built.level_ns, Some(10));
        assert_eq!(rung_ns(&built), Some(40 + CONTRACT_LEVELS * 10));
        assert_eq!(
            class_window(70, 100, &built),
            ClassWindow::Fair { window_ns: 70 },
            "a share that covers the rung is kept"
        );
        assert_eq!(
            class_window(10, 100, &built),
            ClassWindow::Contract {
                window_ns: 60,
                fair_ns: 10
            }
        );
        assert_eq!(
            class_window(10, 59, &built),
            ClassWindow::Decline {
                rung_ns: 60,
                left_ns: 59
            },
            "a rung the tier cannot hold declines (never half-runs)"
        );
        assert_eq!(class_window(10, 59, &built).window_ns(), None);
        // Depth 0 counts as one level; a cut doubles the window it was cut at.
        assert_eq!(none.with_run(5, 7, 0).level_ns, Some(7));
        let cut = none.with_cut(13).with_cut(9);
        assert_eq!(cut.cut_ns, Some(13), "the longest cut window is kept");
        assert_eq!(rung_ns(&cut), Some(26));
        assert_eq!(
            rung_ns(&cut.with_run(4, 2, 1)),
            Some(8),
            "a completed build prices over cuts"
        );
        // Saturation.
        let huge = none.with_run(u64::MAX, u64::MAX, 1);
        assert_eq!(rung_ns(&huge), Some(u64::MAX));
        assert_eq!(rung_ns(&none.with_cut(u64::MAX)), Some(u64::MAX));
        let _ = class_window(u64::MAX, u64::MAX, &huge);
        let _ = class_window(0, 0, &huge);
    }

    /// §6.3: a contained fault disarms the route authority: no capability is
    /// routed and every tier keeps its fair share from there.
    #[test]
    fn router_r5_fault_disarms_the_route_authority() {
        let _serial = globals();
        guard::reset();
        let authority = crate::route_authority::global();
        authority.arm(
            vec![(
                "frozen-dual-bab".to_owned(),
                vec![("k".to_owned(), "true".to_owned())],
            )],
            true,
        );
        assert!(
            authority.is_armed() && authority.gap_contract(),
            "positive control"
        );
        let fault = guard::fail_open("injected-r5", || -> u32 {
            panic!("router-r5 injected panic")
        })
        .expect_err("the injected panic is contained");
        assert!(!fault.skipped);
        assert!(!authority.is_armed());
        assert!(!authority.gap_contract());
        assert!(!authority.routed(super::super::decl::ids::FROZEN_DUAL_BAB));
        guard::reset();
        super::super::bus::reset();
    }

    #[test]
    fn router_r5_arithmetic_never_panics_on_extremes() {
        let mut rng = Rng(0x5eed_5505);
        for _ in 0..300 {
            let mut row = random_input(&mut rng);
            row.facts.params = [None, Some(0), Some(u64::MAX)][rng.below(3) as usize];
            row.facts.p_eff = [None, Some(u64::MAX)][rng.below(2) as usize];
            row.host.threads = [0, u32::MAX][rng.below(2) as usize];
            for cap in &mut row.capabilities {
                let mut spec = sealed_spec("k");
                spec.admit.max_params = Some(rng.next());
                spec.admit.max_p_eff = Some(rng.next());
                spec.admit.min_threads = u32::MAX;
                spec.admit.ops_any = OpSet(rng.next());
                cap.route = routable(&cap.id, LedgerState::Sealed, &spec).route;
            }
            let _ = acting(&row, &["k"]);
            let cost = ClassCost {
                build_ns: rng.coin().then(|| rng.next()),
                level_ns: rng.coin().then(|| rng.next()),
                cut_ns: rng.coin().then(|| rng.next()),
            };
            let _ = class_window(rng.next(), rng.next(), &cost);
            let _ = cost.with_run(rng.next(), rng.next(), rng.next());
            let _ = cost.with_cut(rng.next());
        }
        let a = RouteAuthority::new();
        a.arm(Vec::new(), false);
        let _ = a.snapshot();
    }

    // ---- #admission-census ---------------------------------------------------

    fn unsealed(key: &'static str, slot: Option<&'static str>) -> Spec {
        Spec {
            sr: SrLevel::Decl,
            risk: KnobRisk::High,
            slot,
            ..sealed_spec(key)
        }
    }

    /// The census pretends ONE capability sealed and asks `commit` itself.
    /// A SEALED capability `b` holds the slot both share, so the real commit
    /// routes `b` and refuses the unsealed `a` (`sr:`, `slot:`); the census
    /// takes every other route out, so `a` is admitted on its admission alone
    /// (no slot is ever held in a census), and `b` too. A non-routable or
    /// unregistered id has no decision, and the input is never edited.
    #[test]
    fn census_decision_pretends_only_one_capability_sealed() {
        let open = || LedgerState::Open("S7+S8 empty".to_owned());
        let mut holder = sealed_spec("k.b");
        holder.slot = Some("slot");
        let row = input(
            conv_relu(),
            vec![
                lane(inline()),
                routable("b", LedgerState::Sealed, &holder),
                routable("a", open(), &unsealed("k.a", Some("slot"))),
            ],
        );
        let real = acting(&row, &[]);
        let routed: Vec<&str> = real.routed().map(|d| d.cap.as_str()).collect();
        assert_eq!(routed, ["b"], "the real commit routes the sealed holder");
        let a = real.routes.iter().find(|d| d.cap == "a").expect("decided");
        assert!(
            a.reasons.contains(&"slot:slot:b".to_owned()),
            "{:?}",
            a.reasons
        );
        assert_eq!(route::census_capabilities(&row), ["b", "a"]);
        for id in ["a", "b"] {
            let d = route::census_decision(&row, id, route::CENSUS_WIDEST_ENVELOPE)
                .expect("a routable capability has a census decision");
            assert!(d.routed, "{id}: {:?}", d.reasons);
            assert_eq!(d.cap, id);
            assert_eq!(d.sr, SrLevel::Seal, "pretended sealed");
            assert!(!d.pin_debt);
        }
        assert!(route::census_decision(&row, "lane", route::CENSUS_WIDEST_ENVELOPE).is_none());
        assert!(route::census_decision(&row, "nope", route::CENSUS_WIDEST_ENVELOPE).is_none());
        // The census never edits its input.
        assert_eq!(acting(&row, &[]), real);
    }

    /// A census decision carries ONLY admission reasons, `commit`'s own:
    /// the precondition, the parent's admission and the envelope — never a
    /// seal, moat, pin or slot check.
    #[test]
    fn census_decision_reports_only_admission_reasons() {
        let open = || LedgerState::Open("S7+S8 empty".to_owned());
        let narrow = OpEnvelope {
            ops: OpSet::EMPTY.with(OpKind::Linear),
            specs: ROBUSTNESS,
        };
        let row = input(
            conv_relu(),
            vec![
                lane(inline()),
                routable("a", open(), &unsealed("k.a", None)),
            ],
        );
        let d = route::census_decision(&row, "a", narrow).expect("decided");
        assert_eq!(d.reasons, ["envelope:ops"], "conv and relu lie outside");
        let declined = input(
            conv_relu(),
            vec![
                lane(AdmissionState::Decline("x".to_owned())),
                routable("a", open(), &unsealed("k.a", None)),
            ],
        );
        let d =
            route::census_decision(&declined, "a", route::CENSUS_WIDEST_ENVELOPE).expect("decided");
        assert_eq!(d.reasons, ["parent:lane:decline:x"]);
        let mut halfspace = input(
            conv_relu(),
            vec![
                lane(inline()),
                routable("a", open(), &unsealed("k.a", None)),
            ],
        );
        halfspace.facts.spec = Some(SpecArchetype::Halfspace);
        let d = route::census_decision(&halfspace, "a", route::CENSUS_WIDEST_ENVELOPE)
            .expect("decided");
        assert_eq!(d.reasons, [format!("admit:decline:{DECLINE_SPEC}")]);
        let mut rng = Rng(0xce_5505);
        for _ in 0..200 {
            let mut row = random_input(&mut rng);
            for cap in &mut row.capabilities {
                let spec = if rng.coin() {
                    unsealed("k", Some("slot"))
                } else {
                    sealed_spec("k")
                };
                cap.route = routable(&cap.id, open(), &spec).route;
            }
            let ids: Vec<String> = route::census_capabilities(&row)
                .into_iter()
                .map(str::to_owned)
                .collect();
            for id in ids {
                let d = route::census_decision(&row, &id, route::CENSUS_WIDEST_ENVELOPE)
                    .expect("decided");
                for reason in &d.reasons {
                    assert!(
                        ["admit:", "parent:", "envelope:"]
                            .iter()
                            .any(|p| reason.starts_with(p)),
                        "{id}: a census reason must be an admission check: {reason}"
                    );
                }
                assert_eq!(d.routed, d.reasons.is_empty());
            }
        }
    }

    /// #prepared-root-route: a routable LANE of its own (no parent: its
    /// declaration names its OWN predicate) is routed — and census-admitted —
    /// only where that predicate admits the row: a declined or an unknown
    /// (not inline) own admission is the refusal `own:<token>`, an admitted or
    /// inline one is not. A consumer inside a parent is never asked its own
    /// answer (it follows its parent, `parent:`), so every routable
    /// capability that has a parent decides exactly as before.
    #[test]
    fn census_decision_honours_a_lanes_own_predicate() {
        let open = || LedgerState::Open("S7+S8 empty".to_owned());
        let own = |admission: AdmissionState| {
            let mut spec = unsealed("k.p", None);
            spec.parent = None;
            CapInput {
                admission,
                ..routable("p", open(), &spec)
            }
        };
        for (admission, want) in [
            (AdmissionState::Admit, Vec::<String>::new()),
            (inline(), Vec::new()),
            (
                AdmissionState::Decline("dose".to_owned()),
                vec!["own:decline:dose".to_owned()],
            ),
            (
                AdmissionState::Unknown("no-frozen-levers".to_owned()),
                vec!["own:unknown:no-frozen-levers".to_owned()],
            ),
        ] {
            let row = input(conv_relu(), vec![own(admission.clone())]);
            let d =
                route::census_decision(&row, "p", route::CENSUS_WIDEST_ENVELOPE).expect("decided");
            assert_eq!(d.reasons, want, "{admission:?}");
            assert_eq!(d.routed, want.is_empty());
            // The real commit refuses on the same own answer (beside its seal
            // checks), so the router and the census agree.
            let real = acting(&row, &[]);
            let decided = real.routes.iter().find(|r| r.cap == "p").expect("decided");
            assert_eq!(
                decided.reasons.iter().any(|r| r.starts_with("own:")),
                !want.is_empty(),
                "{:?}",
                decided.reasons
            );
        }
        // A consumer inside an admitted parent is never asked its own answer.
        let child = CapInput {
            admission: AdmissionState::Decline("ignored".to_owned()),
            ..routable("c", open(), &unsealed("k.c", None))
        };
        let row = input(conv_relu(), vec![lane(AdmissionState::Admit), child]);
        let d = route::census_decision(&row, "c", route::CENSUS_WIDEST_ENVELOPE).expect("decided");
        assert!(d.routed, "{:?}", d.reasons);
    }

    /// The census envelope is the widest a seal can record (every kind but
    /// `other`, every archetype), and CR-RS's envelope check is MONOTONE in
    /// it: a row admitted under any narrower envelope is admitted under the
    /// widest, so a moat over the widest census covers every envelope a seal
    /// can derive.
    #[test]
    fn census_widest_envelope_is_every_kind_but_other_and_monotone() {
        let widest = route::CENSUS_WIDEST_ENVELOPE;
        for kind in OpKind::ALL {
            assert_eq!(
                widest.ops.contains(kind),
                !matches!(kind, OpKind::Other),
                "{}",
                kind.token()
            );
        }
        assert_eq!(widest.specs, SpecSet::ALL);
        let open = || LedgerState::Open("S7+S8 empty".to_owned());
        let mut rng = Rng(0xe1_0be5);
        let mut admitted_narrow = 0usize;
        for _ in 0..400 {
            let mut row = random_input(&mut rng);
            let ops = OpSet(rng.next() & widest.ops.0);
            if rng.coin() {
                // Known, complete facts (the rows a narrow envelope can admit).
                let spec = row.facts.spec;
                row.facts = facts(ops);
                row.facts.spec = spec;
            } else {
                row.facts.ops = ops;
            }
            row.capabilities.push(lane(inline()));
            row.capabilities
                .push(routable("c", open(), &unsealed("k", None)));
            // Half the narrow envelopes cover the row's operators (the rest are
            // random subsets of the widest), half carry robustness.
            let extra = OpSet(rng.next() & widest.ops.0);
            let narrow_ops = if rng.coin() {
                OpSet(ops.0 | extra.0)
            } else {
                extra
            };
            let mut narrow_specs = SpecSet(rng.next() as u32 & SpecSet::ALL.0);
            if rng.coin() {
                narrow_specs = narrow_specs.with(SpecArchetype::Robustness);
            }
            let narrow = OpEnvelope {
                ops: narrow_ops,
                specs: narrow_specs,
            };
            let under_narrow = route::census_decision(&row, "c", narrow).expect("decided");
            let under_widest = route::census_decision(&row, "c", widest).expect("decided");
            if under_narrow.routed {
                admitted_narrow += 1;
                assert!(under_widest.routed, "{:?}", under_widest.reasons);
            }
        }
        assert!(
            admitted_narrow > 0,
            "positive control: some narrow envelope admits"
        );
    }
}

// ---------------------------------------------------------------------------
// #router-r6: the O-TRIAL rules (commitments, machines, authority)
// ---------------------------------------------------------------------------

mod r6 {
    use std::time::Duration;

    use super::super::otrial::{
        self, R6Input, TrialCommitments, TrialSwitches, ACTING_SET, R6_RULES,
    };
    use super::super::plan::{
        self, AdmissionState, BudgetView, CapClass, LedgerState, PolicyInput,
    };
    use super::super::trial::{
        start_refusal, AimdTrial, Halving, HalvingRule, Iterate, LrBackoff, LrRule, RestartRule,
        SplitObs, SplitRestart, TrialDecision, TrialDir, TrialEvent, WidthObs, WidthRange,
        WidthRule, TRIAL_LATE_FRAC, TRIAL_WINDOWS_LEFT_MIN, TRIAL_WINDOW_UNITS,
    };
    use super::super::{EffectiveMode, PRIORS_VERSION, ROUTER_SCHEMA};
    use super::{cap, globals, guard, host, random_input, Rng};
    use crate::router::decl::{LaneRole, ProgressKind, SafePoints};
    use crate::router::facts::InstanceFacts;
    use crate::trial_authority::{TrialArmRefusal, TrialAuthority, TrialDirectives, TrialSite};

    const MODES: [EffectiveMode; 3] = [
        EffectiveMode::Off,
        EffectiveMode::Shadow,
        EffectiveMode::Acting,
    ];

    /// A row whose capabilities carry every trial site's lane, each declaring
    /// safe points as the registry declares them.
    fn row(official_ms: u64) -> PolicyInput {
        let lane = |id: &str, role: LaneRole| {
            cap(
                id,
                CapClass::Lane(role),
                LedgerState::SealedAsShipped,
                ProgressKind::Opaque,
                None,
                AdmissionState::Unknown("inline-admission".to_owned()),
            )
        };
        let prover = LaneRole::Prover { terminal_ok: true };
        PolicyInput {
            schema: ROUTER_SCHEMA.to_owned(),
            facts: InstanceFacts::known_empty(),
            host: host(),
            budget: BudgetView {
                official_ms,
                internal_ms: official_ms.saturating_sub(5_000),
                join_margin_ms: 2_000,
            },
            capabilities: vec![
                lane("attack.upfront_apgd", LaneRole::Attack),
                lane("input_split", prover),
                lane("lowdim", prover),
                lane("margin-row", prover),
                lane("wave-beta-bab", prover),
                lane("margin-row.alpha", prover),
            ],
            pins: Vec::new(),
            priors_version: PRIORS_VERSION.to_owned(),
            registry_version: "0123abcd".to_owned(),
            priors: None,
        }
    }

    fn commit(
        input: &PolicyInput,
        mode: EffectiveMode,
        switches: &TrialSwitches,
    ) -> TrialCommitments {
        otrial::commit(
            &plan::plan(input),
            input,
            &R6Input::default(),
            mode,
            switches,
        )
    }

    fn without(id: &str) -> TrialSwitches {
        let mut s = TrialSwitches::ALL_ON;
        match id {
            "rule.trial_width" => s.width = false,
            "rule.trial_lr_backoff" => s.lr_backoff = false,
            "rule.trial_split_restart" => s.split_restart = false,
            "rule.trial_attack_halving" => s.attack_halving = false,
            other => panic!("unknown rule {other}"),
        }
        s
    }

    #[test]
    fn router_r6_trial_commitments_are_pure_and_deterministic() {
        let mut rng = Rng(0x5eed_6601);
        for _ in 0..400 {
            let input = random_input(&mut rng);
            let plan = plan::plan(&input);
            let pins = R6Input {
                pinned_keys: if rng.coin() {
                    vec!["margin_row.tree_wave_batch".to_owned()]
                } else {
                    Vec::new()
                },
            };
            for mode in MODES {
                let switches = TrialSwitches {
                    width: rng.coin(),
                    lr_backoff: rng.coin(),
                    split_restart: rng.coin(),
                    attack_halving: rng.coin(),
                };
                let a = otrial::commit(&plan, &input, &pins, mode, &switches);
                let b = otrial::commit(&plan, &input, &pins, mode, &switches);
                assert_eq!(a, b, "commit is a pure function of its input");
                assert_eq!(a.sha.len(), 64);
                assert_eq!(a.rules.len(), R6_RULES.len(), "every rule decides");
                assert_eq!(a.sites.len(), TrialSite::ALL.len(), "every site decides");
                for (act, id) in a.rules.iter().zip(R6_RULES) {
                    assert_eq!(act.rule, id.as_str());
                    assert_eq!(act.acts, act.blocked_by.is_none(), "{act:?}");
                }
                for site in &a.sites {
                    assert_eq!(site.armed, site.blocked_by.is_none(), "{site:?}");
                    assert_eq!(
                        site.armed,
                        a.directives.site_armed(site.site),
                        "the directives arm exactly the armed sites: {site:?}"
                    );
                }
                // A rule acts iff one of its sites is armed.
                for id in R6_RULES {
                    let any_site = a.sites.iter().any(|s| s.rule == id.as_str() && s.armed);
                    assert_eq!(a.acts(id), any_site, "{id:?}");
                }
            }
        }
        assert_eq!(
            &ACTING_SET[4..],
            &R6_RULES,
            "R6 follows R3 in the acting set"
        );
    }

    #[test]
    fn router_r6_nothing_trials_unless_the_optimizer_acts() {
        let input = row(100_000);
        for mode in [EffectiveMode::Off, EffectiveMode::Shadow] {
            let c = commit(&input, mode, &TrialSwitches::ALL_ON);
            assert!(!c.acting);
            assert_eq!(c.directives, TrialDirectives::NONE, "{mode:?} arms nothing");
            for site in &c.sites {
                assert_eq!(
                    site.blocked_by.as_deref(),
                    Some(format!("mode:{}", mode.token()).as_str())
                );
            }
        }
        let on = commit(&input, EffectiveMode::Acting, &TrialSwitches::ALL_ON);
        assert!(on.acting, "positive control: `on` acts");
        assert!(R6_RULES.iter().all(|id| on.acts(*id)), "{:?}", on.rules);
        for site in TrialSite::ALL {
            assert!(on.directives.site_armed(site), "{site:?}");
        }
        assert_eq!(on.directives.width, Some(WidthRule::SHIPPED));
        assert_eq!(on.directives.lr_backoff, Some(LrRule::SHIPPED));
        assert_eq!(on.directives.split_restart, Some(RestartRule::SHIPPED));
        assert_eq!(on.directives.attack_halving, Some(HalvingRule::SHIPPED));
    }

    #[test]
    fn router_r6_each_rule_key_ablates_its_own_rule() {
        let input = row(100_000);
        for id in R6_RULES {
            let c = commit(&input, EffectiveMode::Acting, &without(id.as_str()));
            assert!(!c.acts(id), "{id:?}");
            for site in c.sites.iter().filter(|s| s.rule == id.as_str()) {
                assert_eq!(site.blocked_by.as_deref(), Some("key"), "{site:?}");
            }
            for other in R6_RULES.iter().filter(|o| **o != id) {
                assert!(
                    c.acts(*other),
                    "ablating {id:?} must leave {other:?} acting"
                );
            }
        }
    }

    /// CR-S: a row of 30 s or less takes no active probing, and a trial is
    /// one.
    #[test]
    fn router_r6_short_row_trials_nothing() {
        let short = commit(&row(20_000), EffectiveMode::Acting, &TrialSwitches::ALL_ON);
        assert!(!short.directives.any());
        for site in &short.sites {
            assert_eq!(site.blocked_by.as_deref(), Some("short_row"), "{site:?}");
        }
        let long = commit(&row(31_000), EffectiveMode::Acting, &TrialSwitches::ALL_ON);
        assert!(long.directives.any(), "positive control");
    }

    /// A trial changes a value only at a declared safe point: a site whose
    /// capability declares none (or is absent) is never armed.
    #[test]
    fn router_r6_site_without_safe_points_is_blocked() {
        let mut input = row(100_000);
        for c in &mut input.capabilities {
            if c.id == "wave-beta-bab" {
                c.safe_points = SafePoints::None;
            }
        }
        input.capabilities.retain(|c| c.id != "lowdim");
        let c = commit(&input, EffectiveMode::Acting, &TrialSwitches::ALL_ON);
        for (site, blocked) in [
            (TrialSite::WaveWidth, true),
            (TrialSite::LowdimWidth, true),
            (TrialSite::LowdimSplit, true),
            (TrialSite::InputSplitWidth, false),
            (TrialSite::AlphaLr, false),
            (TrialSite::AttackStep, false),
        ] {
            let act = c.site(site).expect("site decided");
            if blocked {
                assert_eq!(act.blocked_by.as_deref(), Some("safe_points"), "{act:?}");
            } else {
                assert!(act.armed, "{act:?}");
            }
        }
        assert!(
            c.acts(otrial::RULE_TRIAL_WIDTH),
            "the input-split site still arms the rule"
        );
        assert!(!c.acts(otrial::RULE_TRIAL_SPLIT_RESTART));
    }

    /// Review fix (router-r6): a trial never tunes a lane whose ledger is not
    /// SEALED (§5.3 / §6.2; R2's `ledger:<id>` refusal). The registry's wave
    /// tree (`wave-beta-bab`) is `Open` until M6 seals it, so its width site
    /// is blocked `ledger:wave-beta-bab` even under `optimizer.mode: on`; a
    /// `Shadow` lane is blocked the same way; every sealed site still arms.
    #[test]
    fn router_r6_site_of_an_unsealed_lane_is_blocked() {
        for ledger in [LedgerState::Open("S7".to_owned()), LedgerState::Shadow] {
            let mut input = row(100_000);
            for c in &mut input.capabilities {
                if c.id == "wave-beta-bab" || c.id == "lowdim" {
                    c.ledger = ledger.clone();
                }
            }
            let c = commit(&input, EffectiveMode::Acting, &TrialSwitches::ALL_ON);
            for (site, blocked) in [
                (TrialSite::WaveWidth, Some("ledger:wave-beta-bab")),
                (TrialSite::LowdimWidth, Some("ledger:lowdim")),
                (TrialSite::LowdimSplit, Some("ledger:lowdim")),
                (TrialSite::InputSplitWidth, None),
                (TrialSite::AlphaLr, None),
                (TrialSite::AttackStep, None),
            ] {
                let act = c.site(site).expect("site decided");
                assert_eq!(act.blocked_by.as_deref(), blocked, "{ledger:?} {act:?}");
                assert_eq!(c.directives.site_armed(site), blocked.is_none(), "{act:?}");
            }
            assert!(
                c.acts(otrial::RULE_TRIAL_WIDTH),
                "input_split still arms it"
            );
            assert!(
                !c.acts(otrial::RULE_TRIAL_SPLIT_RESTART),
                "its only site is unsealed"
            );
            assert_eq!(
                c.rules
                    .iter()
                    .find(|r| r.rule == otrial::RULE_TRIAL_SPLIT_RESTART.as_str())
                    .and_then(|r| r.blocked_by.as_deref()),
                Some("ledger:lowdim")
            );
        }
        // Positive control: every site of the all-sealed row arms.
        let sealed = commit(&row(100_000), EffectiveMode::Acting, &TrialSwitches::ALL_ON);
        assert!(TrialSite::ALL
            .iter()
            .all(|site| sealed.directives.site_armed(*site)));
    }

    /// A pinned knob is FROZEN: its site is not armed, under every path the
    /// knob may be written at; sites of code constants are never pinned.
    #[test]
    fn router_r6_pinned_knob_freezes_its_site() {
        let input = row(100_000);
        let plan = plan::plan(&input);
        for (path, site) in [
            ("margin_row.tree_wave_batch", TrialSite::WaveWidth),
            ("bab.batch_size", TrialSite::InputSplitWidth),
            ("solver.batch_size", TrialSite::InputSplitWidth),
            ("margin_row.alpha_lr", TrialSite::AlphaLr),
            (
                "bab.branching.input_split.lowdim_split",
                TrialSite::LowdimSplit,
            ),
        ] {
            let c = otrial::commit(
                &plan,
                &input,
                &R6Input {
                    pinned_keys: vec![path.to_owned()],
                },
                EffectiveMode::Acting,
                &TrialSwitches::ALL_ON,
            );
            let act = c.site(site).expect("site");
            assert_eq!(
                act.blocked_by.as_deref(),
                Some(format!("pin:{path}").as_str()),
                "{act:?}"
            );
            assert!(!c.directives.site_armed(site));
            let others = TrialSite::ALL.iter().filter(|s| **s != site);
            for other in others {
                assert!(c.directives.site_armed(*other), "{path} pins only {site:?}");
            }
        }
        assert_eq!(TrialSite::LowdimWidth.pin_paths().len(), 0);
        assert_eq!(TrialSite::AttackStep.pin_paths().len(), 0);
    }

    /// The §5.1 / §5.3 protocol gate: >= 16 units per window, >= 8 windows
    /// left, never in the last 25%, <= 2 trials per knob, a known deadline.
    #[test]
    fn router_r6_trial_protocol_preconditions() {
        let ok = |units, secs, elapsed, remaining, used| {
            start_refusal(
                units,
                secs,
                elapsed,
                remaining,
                used,
                2,
                TRIAL_WINDOWS_LEFT_MIN,
                TRIAL_LATE_FRAC,
            )
        };
        assert_eq!(ok(16, 1.0, 10.0, Some(80.0), 0), None);
        assert_eq!(ok(15, 1.0, 10.0, Some(80.0), 0), Some("window_units"));
        assert_eq!(ok(16, 1.0, 10.0, Some(7.9), 0), Some("windows_left"));
        assert_eq!(ok(16, 1.0, 10.0, None, 0), Some("deadline_unknown"));
        assert_eq!(ok(16, 0.0, 10.0, Some(80.0), 0), Some("window_secs"));
        assert_eq!(ok(16, 1.0, 10.0, Some(80.0), 2), Some("trials_exhausted"));
        // Exactly 75% elapsed is late (never in the last quarter).
        assert_eq!(ok(16, 0.1, 75.0, Some(25.0), 0), Some("late"));
        assert_eq!(ok(16, 0.1, 74.0, Some(26.0), 0), None);
        assert_eq!(TRIAL_WINDOW_UNITS, 16);
    }

    fn width(base: u64) -> AimdTrial {
        let range = WidthRange::around(base, 4, 1, u64::MAX).expect("range");
        AimdTrial::new(
            WidthRule::SHIPPED,
            "rule.trial_width",
            "lowdim",
            "lowdim.batch",
            base,
            range,
        )
        .expect("machine")
    }

    /// Feed one window of 16 units at `rate` closes/s (1 s per window), with
    /// a far deadline. Returns the events and advances the stream state.
    fn window(
        m: &mut AimdTrial,
        st: &mut (u64, u64, f64),
        rate: f64,
        lease: Option<f64>,
    ) -> Vec<TrialEvent> {
        let mut out = Vec::new();
        for _ in 0..16 {
            st.0 += 1;
            st.2 += 1.0 / 16.0;
            st.1 += (rate / 16.0).round() as u64;
            out.extend(m.observe(WidthObs {
                units: st.0,
                closes: st.1,
                t_s: st.2,
                remaining_s: Some(1_000.0),
                lease_use: lease,
            }));
        }
        out
    }

    fn primed(base: u64) -> (AimdTrial, (u64, u64, f64)) {
        let mut m = width(base);
        let mut st = (0u64, 0u64, 0.0f64);
        // The first observation sets the mark; then one full window.
        st.0 += 1;
        assert_eq!(
            m.observe(WidthObs {
                units: st.0,
                closes: 0,
                t_s: 0.0,
                remaining_s: Some(1_000.0),
                lease_use: Some(0.1),
            })
            .len(),
            0
        );
        assert!(
            window(&mut m, &mut st, 1_600.0, Some(0.1)).is_empty(),
            "no prev yet"
        );
        (m, st)
    }

    #[test]
    fn router_r6_aimd_raises_on_a_rise_and_keeps_a_better_window() {
        let (mut m, mut st) = primed(256);
        let ev = window(&mut m, &mut st, 1_760.0, Some(0.1)); // +10%
        assert_eq!(ev.len(), 1, "{ev:?}");
        assert_eq!(
            (ev[0].dir, ev[0].decision),
            (TrialDir::Up, TrialDecision::Started)
        );
        assert_eq!(m.value(), 512, "x2");
        let ev = window(&mut m, &mut st, 1_920.0, Some(0.1)); // >= baseline x 1.05
        assert_eq!(ev[0].decision, TrialDecision::Kept, "{ev:?}");
        assert_eq!(m.value(), 512);
        assert!(ev[0]
            .line()
            .starts_with("[router] trial knob=lowdim.batch dir=up decision=kept"));
    }

    #[test]
    fn router_r6_aimd_two_falls_after_a_raise_back_to_last_good_and_freeze() {
        let (mut m, mut st) = primed(256);
        window(&mut m, &mut st, 1_760.0, Some(0.1));
        assert_eq!(m.value(), 512);
        assert!(
            window(&mut m, &mut st, 1_600.0, Some(0.1)).is_empty(),
            "first fall: wait"
        );
        let ev = window(&mut m, &mut st, 1_500.0, Some(0.1));
        let decisions: Vec<TrialDecision> = ev.iter().map(|e| e.decision).collect();
        assert_eq!(
            decisions,
            vec![TrialDecision::Reverted, TrialDecision::Frozen],
            "{ev:?}"
        );
        assert_eq!(m.value(), 256, "back to the last good width");
        assert!(m.frozen());
        // Frozen: nothing moves again, whatever the stream does.
        for rate in [3_000.0, 100.0, 5_000.0, 10.0] {
            assert_eq!(window(&mut m, &mut st, rate, Some(0.1)).len(), 0);
            assert_eq!(m.value(), 256);
        }
    }

    #[test]
    fn router_r6_aimd_revert_makes_the_direction_tabu() {
        let (mut m, mut st) = primed(256);
        window(&mut m, &mut st, 1_760.0, Some(0.1)); // raise
                                                     // Not kept (below baseline x 1.05) but not two falls either.
        assert_eq!(window(&mut m, &mut st, 1_780.0, Some(0.1)).len(), 0);
        let ev = window(&mut m, &mut st, 1_790.0, Some(0.1));
        assert_eq!(ev[0].decision, TrialDecision::Reverted, "{ev:?}");
        assert_eq!(ev[0].why, "not_kept");
        assert_eq!(m.value(), 256);
        assert!(m.tabu(TrialDir::Up) && !m.frozen());
        // Another rise: the direction is tabu (reported once), nothing moves.
        let ev = window(&mut m, &mut st, 2_400.0, Some(0.1));
        assert_eq!(ev.len(), 1, "{ev:?}");
        assert_eq!(ev[0].decision, TrialDecision::Tabu);
        assert_eq!(m.value(), 256);
        assert!(
            window(&mut m, &mut st, 3_000.0, Some(0.1)).is_empty(),
            "reported once"
        );
        // A fall may still trial the other direction.
        let ev = window(&mut m, &mut st, 2_000.0, Some(0.1));
        assert_eq!(
            (ev[0].dir, ev[0].decision),
            (TrialDir::Down, TrialDecision::Started)
        );
        assert_eq!(m.value(), 128);
    }

    #[test]
    fn router_r6_aimd_lease_guard_reverts_a_raise_and_halves() {
        let (mut m, mut st) = primed(256);
        // A rise with lease >= 70% does not raise.
        assert_eq!(window(&mut m, &mut st, 1_760.0, Some(0.75)).len(), 0);
        assert_eq!(m.value(), 256);
        // A rise under 70% raises; the next window's lease > 85% reverts it
        // at once (guard), whatever the metric.
        window(&mut m, &mut st, 2_000.0, Some(0.5));
        assert_eq!(m.value(), 512);
        let ev = window(&mut m, &mut st, 9_000.0, Some(0.9));
        assert_eq!(ev[0].decision, TrialDecision::Reverted);
        assert_eq!(ev[0].why, "lease_guard");
        assert_eq!(m.value(), 256);
        // Review fix (router-r6): with nothing pending, lease > 85% STARTS a
        // halving TRIAL (the §5.2 "/2 on ... or lease > 85%" trigger) through
        // the protocol gate — counted against the knob's budget, never an
        // uncounted guard.
        let used = m.trials_used();
        let ev = window(&mut m, &mut st, 9_000.0, Some(0.95));
        assert_eq!(
            (ev[0].dir, ev[0].decision, ev[0].why.as_str()),
            (TrialDir::Down, TrialDecision::Started, "lease"),
            "{ev:?}"
        );
        assert_eq!(m.value(), 128);
        assert_eq!(m.trials_used(), used + 1, "lease pressure starts a TRIAL");
        // No machine emits the retired out-of-protocol guard any more.
        assert!(ev.iter().all(|e| e.decision != TrialDecision::Guard));
        // An unknown lease never raises.
        let (mut m2, mut st2) = primed(256);
        let ev = window(&mut m2, &mut st2, 2_400.0, None);
        assert_eq!(ev[0].decision, TrialDecision::Blocked);
        assert_eq!(ev[0].why, "lease_unknown");
        assert_eq!(m2.value(), 256);
    }

    /// Review fix (router-r6): memory pressure moves NOTHING outside the
    /// protocol. Late in the lane (past 75%) lease > 85% is `blocked late`
    /// and today's value stays (the old guard halved it there, uncounted,
    /// down to the range floor window after window); with time left it starts
    /// at most one halving trial, judged like any other and reverted (tabu)
    /// when it does not beat its baseline.
    #[test]
    fn router_r6_aimd_lease_pressure_never_moves_today_outside_the_protocol() {
        let late = |m: &mut AimdTrial, st: &mut (u64, u64, f64)| {
            let mut out = Vec::new();
            for _ in 0..16 {
                st.0 += 1;
                st.2 += 1.0 / 16.0;
                st.1 += 100;
                out.extend(m.observe(WidthObs {
                    units: st.0,
                    closes: st.1,
                    t_s: 1_000.0 + st.2,
                    // 1 000 s spent, 100 s left: past 75% of the lane.
                    remaining_s: Some(100.0),
                    lease_use: Some(0.99),
                }));
            }
            out
        };
        let (mut m, mut st) = primed(256);
        let mut refusals = Vec::new();
        for _ in 0..6 {
            let ev = late(&mut m, &mut st);
            assert!(
                ev.iter().all(|e| e.decision == TrialDecision::Blocked),
                "{ev:?}"
            );
            assert_eq!(m.value(), 256, "today's value, late in the lane");
            refusals.extend(ev);
        }
        assert!(
            refusals
                .iter()
                .any(|e| e.why == "late" && e.dir == TrialDir::Down),
            "pressure late in the lane is refused `late`: {refusals:?}"
        );
        assert_eq!(m.trials_used(), 0);
        // With time left: ONE halving trial, judged; a flat metric reverts it
        // and makes `down` tabu, so pressure cannot ratchet the width down.
        let (mut m, mut st) = primed(256);
        let ev = window(&mut m, &mut st, 1_600.0, Some(0.99));
        assert_eq!(ev[0].decision, TrialDecision::Started, "{ev:?}");
        assert_eq!(m.value(), 128);
        let mut all = Vec::new();
        for _ in 0..6 {
            all.extend(window(&mut m, &mut st, 1_600.0, Some(0.99)));
        }
        assert!(
            all.iter()
                .any(|e| e.decision == TrialDecision::Reverted && e.why == "not_kept"),
            "{all:?}"
        );
        assert_eq!(m.value(), 256, "reverted to today's value");
        assert!(m.tabu(TrialDir::Down));
        assert_eq!(m.trials_used(), 1, "one trial, never an uncounted halving");
    }

    /// <= 2 trials per knob per row, then a `blocked` record (once).
    #[test]
    fn router_r6_aimd_at_most_two_trials_per_knob() {
        let (mut m, mut st) = primed(256);
        let mut rate = 1_600.0;
        let mut started = 0;
        for _ in 0..40 {
            rate *= 1.2; // every window rises: raise, keep, raise, keep, ...
            for e in window(&mut m, &mut st, rate, Some(0.1)) {
                if e.decision == TrialDecision::Started {
                    started += 1;
                }
            }
        }
        assert_eq!(started, 2, "two trials, never a third");
        assert_eq!(m.trials_used(), 2);
        assert!(m.value() <= m.range().max);
    }

    /// Property: on random streams the value never leaves its declared range,
    /// never moves while frozen, and the machine never panics.
    #[test]
    fn router_r6_aimd_never_leaves_its_range() {
        let mut rng = Rng(0x5eed_6602);
        for _ in 0..300 {
            let base = 1 + rng.below(4096);
            let range = WidthRange::around(base, 4, 1 + rng.below(4), 1 + rng.below(1 << 14));
            let Some(range) = range else {
                continue;
            };
            let mut m =
                AimdTrial::new(WidthRule::SHIPPED, "r", "s", "k", base, range).expect("machine");
            let (mut units, mut closes, mut t) = (0u64, 0u64, 0.0f64);
            let mut frozen_at: Option<u64> = None;
            for _ in 0..400 {
                units += 1 + rng.below(3);
                closes += rng.below(50);
                t += [0.0, 0.01, 0.5, f64::NAN][rng.below(4) as usize];
                let lease =
                    [None, Some(0.0), Some(0.8), Some(0.9), Some(f64::NAN)][rng.below(5) as usize];
                let remaining = [None, Some(1.0), Some(1e6)][rng.below(3) as usize];
                let _ = m.observe(WidthObs {
                    units,
                    closes,
                    t_s: t,
                    remaining_s: remaining,
                    lease_use: lease,
                });
                assert!(range.contains(m.value()), "{} not in {range:?}", m.value());
                if let Some(v) = frozen_at {
                    assert_eq!(m.value(), v, "frozen means frozen");
                } else if m.frozen() {
                    frozen_at = Some(m.value());
                }
                assert!(m.trials_used() <= 2);
            }
        }
    }

    /// The site range: `[base / 4, base x 4]` inside the site's own clamp;
    /// `None` when the clamp excludes today's value.
    #[test]
    fn router_r6_width_range_stays_inside_the_hard_clamp() {
        assert_eq!(
            WidthRange::around(256, 4, 64, 1024),
            Some(WidthRange { min: 64, max: 1024 })
        );
        assert_eq!(
            WidthRange::around(2048, 4, 1, 1 << 16),
            Some(WidthRange {
                min: 512,
                max: 8192
            })
        );
        assert_eq!(
            WidthRange::around(1024, 4, 64, 1024),
            Some(WidthRange {
                min: 256,
                max: 1024
            })
        );
        assert_eq!(
            WidthRange::around(2048, 4, 64, 1024),
            None,
            "base past the clamp"
        );
        assert_eq!(
            WidthRange::around(1, 4, 1, 1),
            Some(WidthRange { min: 1, max: 1 })
        );
        assert_eq!(
            WidthRange::around(u64::MAX, 4, 1, u64::MAX).map(|r| r.max),
            Some(u64::MAX)
        );
    }

    fn lr() -> LrBackoff {
        LrBackoff::new(
            LrRule::SHIPPED,
            "rule.trial_lr_backoff",
            "margin-row.alpha",
            "margin_row.alpha_lr",
        )
        .expect("machine")
    }

    fn it(index: u64, accepted: bool, gain: f64) -> Iterate {
        Iterate {
            index,
            accepted,
            gain,
            secs: 0.5,
            done_frac: index as f64 / 40.0,
        }
    }

    #[test]
    fn router_r6_lr_backoff_halves_after_three_rejections_and_floors() {
        let mut m = lr();
        assert_eq!(m.scale(), 1.0);
        assert_eq!(m.iterate(it(1, true, 0.01)).len(), 0);
        assert_eq!(m.iterate(it(2, false, -0.01)).len(), 0);
        assert_eq!(m.iterate(it(3, false, -0.01)).len(), 0);
        let ev = m.iterate(it(4, false, -0.01));
        assert_eq!(
            (ev[0].dir, ev[0].decision),
            (TrialDir::Down, TrialDecision::Started)
        );
        assert_eq!(m.scale(), 0.5, "x 0.5");
        // The next three iterates hold the gain rate: kept.
        m.iterate(it(5, false, -0.001));
        m.iterate(it(6, false, -0.001));
        let ev = m.iterate(it(7, false, -0.001));
        assert_eq!(ev[0].decision, TrialDecision::Kept, "{ev:?}");
        // A second backoff, then the trial budget is spent: the step never
        // goes below lr0 / 8 and never above lr0.
        for i in 8..11 {
            m.iterate(it(i, false, -0.001));
        }
        assert_eq!(m.scale(), 0.25);
        for i in 11..30 {
            m.iterate(it(i, false, -0.0001));
            assert!(m.scale() >= LrRule::SHIPPED.floor() && m.scale() <= 1.0);
        }
        assert_eq!(m.trials_used(), 2);
        assert!(m.scale() >= 1.0 / 8.0);
    }

    #[test]
    fn router_r6_lr_backoff_reverts_when_gain_rate_falls_and_becomes_tabu() {
        let mut m = lr();
        for i in 1..=3 {
            m.iterate(it(i, false, -0.01));
        }
        assert_eq!(m.scale(), 0.5);
        m.iterate(it(4, false, -0.05));
        m.iterate(it(5, false, -0.05));
        let ev = m.iterate(it(6, false, -0.05));
        assert_eq!(ev[0].decision, TrialDecision::Reverted, "{ev:?}");
        assert_eq!(ev[0].why, "gain_rate_fell");
        assert_eq!(m.scale(), 1.0, "back to today's step");
        assert!(m.tabu());
        // Three more rejections: tabu (reported once), the step stays.
        let mut tabu = 0;
        for i in 7..20 {
            for e in m.iterate(it(i, false, -0.01)) {
                assert_eq!(e.decision, TrialDecision::Tabu);
                tabu += 1;
            }
            assert_eq!(m.scale(), 1.0);
        }
        assert_eq!(tabu, 1);
    }

    /// Property: whatever the stream, the multiplier is never above 1 (no
    /// raise, ever) and never below the floor; and never starts late.
    #[test]
    fn router_r6_lr_backoff_never_raises() {
        let mut rng = Rng(0x5eed_6603);
        for _ in 0..300 {
            let mut m = lr();
            for i in 0..60 {
                let frac = rng.below(100) as f64 / 100.0;
                let ev = m.iterate(Iterate {
                    index: i,
                    accepted: rng.coin(),
                    gain: [-1.0, 0.0, 1e-9, f64::NAN, f64::NEG_INFINITY][rng.below(5) as usize],
                    secs: [0.0, 0.1, f64::INFINITY][rng.below(3) as usize],
                    done_frac: frac,
                });
                for e in &ev {
                    if e.decision == TrialDecision::Started {
                        assert!(frac < TRIAL_LATE_FRAC, "never starts late");
                    }
                    assert!(e.metric.is_finite() && e.baseline.is_finite());
                }
                assert!(m.scale() <= 1.0 && m.scale() >= LrRule::SHIPPED.floor());
            }
        }
    }

    fn split() -> SplitRestart {
        SplitRestart::new(
            RestartRule::SHIPPED,
            "rule.trial_split_restart",
            "lowdim",
            "bab.branching.input_split.lowdim_split",
            "widest",
            "sbwr",
        )
        .expect("machine")
    }

    fn sobs(units: u64, vol: f64, open: u64, children: u64, zero: u64, t: f64) -> SplitObs {
        SplitObs {
            units,
            closed_volume: vol,
            open,
            children,
            zero_lift: zero,
            t_s: t,
            remaining_s: Some(1_000.0),
        }
    }

    #[test]
    fn router_r6_split_restart_fires_once_on_far_and_zero_lift() {
        let mut m = split();
        assert!(m.observe(sobs(0, 0.0, 1, 0, 0, 0.0)).is_none());
        assert!(
            m.observe(sobs(16, 0.0, 10, 30, 29, 1.0)).is_none(),
            "one window: no evidence"
        );
        let ev = m.observe(sobs(32, 0.0, 20, 60, 58, 2.0)).expect("restart");
        assert_eq!(
            (ev.dir, ev.decision),
            (TrialDir::Switch, TrialDecision::Started)
        );
        assert!(ev.why.contains("from=widest to=sbwr"), "{ev:?}");
        assert!(m.used());
        assert_eq!(m.target(), "sbwr");
        // Once per row: a FAR, zero-lift stream never restarts again.
        for k in 3..20u64 {
            assert!(m
                .observe(sobs(16 * k, 0.0, 20 * k, 30 * k, 30 * k, k as f64))
                .is_none());
        }
    }

    #[test]
    fn router_r6_split_restart_needs_both_far_and_zero_lift() {
        // FAR but lifting splits: no restart.
        let mut m = split();
        m.observe(sobs(0, 0.0, 1, 0, 0, 0.0));
        m.observe(sobs(16, 0.0, 10, 30, 15, 1.0));
        assert!(m.observe(sobs(32, 0.0, 20, 60, 30, 2.0)).is_none());
        // Zero-lift but NOT far (volume grows): no restart.
        let mut m = split();
        m.observe(sobs(0, 0.0, 1, 0, 0, 0.0));
        m.observe(sobs(16, 0.1, 10, 30, 30, 1.0));
        assert!(m.observe(sobs(32, 0.2, 20, 60, 60, 2.0)).is_none());
        // Zero-lift but the queue shrinks (not FAR): no restart.
        let mut m = split();
        m.observe(sobs(0, 0.0, 30, 0, 0, 0.0));
        m.observe(sobs(16, 0.0, 20, 30, 30, 1.0));
        assert!(m.observe(sobs(32, 0.0, 10, 60, 60, 2.0)).is_none());
        // Late in the lane: refused (blocked once), never a restart.
        let mut m = split();
        m.observe(sobs(0, 0.0, 1, 0, 0, 900.0));
        m.observe(sobs(16, 0.0, 10, 30, 29, 3_000.0));
        let ev = m
            .observe(sobs(32, 0.0, 20, 60, 58, 3_100.0))
            .expect("blocked record");
        assert_eq!(ev.decision, TrialDecision::Blocked);
        assert!(!m.used());
        assert!(
            SplitRestart::new(RestartRule::SHIPPED, "r", "s", "k", "widest", "widest").is_none()
        );
    }

    fn halving() -> Halving {
        Halving::new(
            HalvingRule::SHIPPED,
            "rule.trial_attack_halving",
            "attack.upfront_apgd",
            "attack.upfront_apgd.step0",
        )
        .expect("machine")
    }

    /// Restart 0 keeps today's step; each round gives every surviving arm the
    /// same number of restarts with both parities; the better half by
    /// improvement per second survives; one arm is left after two rounds.
    #[test]
    fn router_r6_successive_halving_keeps_the_better_half() {
        let mut m = halving();
        let rate = |step: f64| match step {
            s if (s - 0.5).abs() < 1e-12 => 5.0,
            s if (s - 0.25).abs() < 1e-12 => 3.0,
            s if (s - 0.125).abs() < 1e-12 => 1.0,
            _ => 0.5,
        };
        let mut seen: Vec<(usize, f64)> = Vec::new();
        let mut events = Vec::new();
        for restart in 0..40u64 {
            let (arm, step) = m.next_arm();
            if restart == 0 {
                assert_eq!(arm, 0);
                assert!((step - 0.25).abs() < 1e-12, "restart 0 is today's restart");
            }
            seen.push((arm, step));
            events.extend(m.done_restart(arm, rate(step), 1.0, restart as f64 / 100.0));
        }
        let kept: Vec<&TrialEvent> = events
            .iter()
            .filter(|e| e.decision == TrialDecision::Kept)
            .collect();
        assert_eq!(kept.len(), 2, "{events:?}");
        assert_eq!((kept[0].from, kept[0].to), (4.0, 2.0));
        assert!(
            kept[0].why.starts_with("kept=0.25,0.5 "),
            "{:?}",
            kept[0].why
        );
        assert_eq!((kept[1].from, kept[1].to), (2.0, 1.0));
        assert_eq!(m.alive_steps(), vec![0.5]);
        // Round 0 is restarts 0..8: every arm twice, once on each parity.
        for arm in 0..4 {
            let parities: Vec<usize> = seen[..8]
                .iter()
                .enumerate()
                .filter(|(_, (a, _))| *a == arm)
                .map(|(i, _)| i % 2)
                .collect();
            assert_eq!(parities.len(), 2, "arm {arm}");
            assert_ne!(parities[0], parities[1], "arm {arm} sees both parities");
        }
        // After the last elimination every restart takes the winner.
        assert!(seen[16..].iter().all(|(_, s)| (*s - 0.5).abs() < 1e-12));
    }

    /// Review fix (router-r6): past 75% of the lane the machine eliminates
    /// nothing AND tries no untested step. It freezes at the FIRST restart
    /// that reports late (not only at a round's end), and from then every
    /// restart takes its leader: today's step (arm 0) while no elimination
    /// has been decided, the leading survivor after one. (It used to keep
    /// handing out the round's untested steps and then rotate over every
    /// survivor in the last quarter.)
    #[test]
    fn router_r6_successive_halving_freezes_late() {
        let mut m = halving();
        let mut events = Vec::new();
        let (arm, step) = m.next_arm();
        assert_eq!((arm, step), (0, 0.25));
        events.extend(m.done_restart(arm, 1.0, 1.0, 0.9));
        assert_eq!(events.len(), 1, "{events:?}");
        assert_eq!(events[0].decision, TrialDecision::Frozen);
        assert!(events[0].why.contains("leader=0.25"), "{events:?}");
        assert!(m.frozen(), "frozen at the first late restart");
        assert_eq!(m.alive_steps().len(), 4, "no elimination late in the lane");
        // Frozen before any elimination: every restart is today's step.
        for _ in 0..12 {
            let (arm, step) = m.next_arm();
            assert_eq!((arm, step), (0, 0.25), "no untested step late");
            assert_eq!(m.done_restart(arm, 9.0, 1.0, 0.95).len(), 0);
        }
        // After one elimination round (early), late: the leading survivor.
        let mut m = halving();
        let rate = |step: f64| if (step - 0.5).abs() < 1e-12 { 5.0 } else { 1.0 };
        for _ in 0..8 {
            let (arm, step) = m.next_arm();
            m.done_restart(arm, rate(step), 1.0, 0.1);
        }
        assert_eq!(m.alive_steps(), vec![0.25, 0.5], "round 0 eliminated two");
        let (arm, step) = m.next_arm();
        let ev = m.done_restart(arm, rate(step), 1.0, 0.8);
        assert_eq!(ev[0].decision, TrialDecision::Frozen, "{ev:?}");
        for _ in 0..8 {
            assert_eq!(m.next_arm().1, 0.5, "the leader takes every late restart");
        }
    }

    #[test]
    fn router_r6_trial_authority_unarmed_answers_nothing() {
        let a = TrialAuthority::new();
        assert!(!a.is_armed());
        for site in TrialSite::ALL {
            assert!(!a.site_armed(site));
            assert_eq!(a.width(site), None);
        }
        assert_eq!(a.lr_backoff(), None);
        assert_eq!(a.split_restart(), None);
        assert_eq!(a.attack_halving(), None);
        a.emit(
            vec![width(4).observe(WidthObs {
                units: 1,
                closes: 0,
                t_s: 0.0,
                remaining_s: None,
                lease_use: None,
            })]
            .into_iter()
            .flatten()
            .collect(),
        );
        assert_eq!(a.snapshot().events.len(), 0);
        a.arm(TrialDirectives::NONE)
            .expect("empty directives are valid");
        assert!(!a.is_armed(), "arming nothing arms nothing");
    }

    #[test]
    fn router_r6_trial_authority_refuses_invalid_rules() {
        let a = TrialAuthority::new();
        let bad_width = WidthRule {
            window_units: 8,
            ..WidthRule::SHIPPED
        };
        let bad_late = WidthRule {
            late_frac: 0.9,
            ..WidthRule::SHIPPED
        };
        let bad_trials = WidthRule {
            max_trials: 3,
            ..WidthRule::SHIPPED
        };
        for bad in [bad_width, bad_late, bad_trials] {
            assert_eq!(
                a.arm(TrialDirectives {
                    width: Some(bad),
                    sites: TrialSite::LowdimWidth.bit(),
                    ..TrialDirectives::NONE
                }),
                Err(TrialArmRefusal::RuleOutOfRange)
            );
        }
        let raising = LrRule {
            factor: 2.0,
            ..LrRule::SHIPPED
        };
        assert_eq!(
            a.arm(TrialDirectives {
                lr_backoff: Some(raising),
                sites: TrialSite::AlphaLr.bit(),
                ..TrialDirectives::NONE
            }),
            Err(TrialArmRefusal::RuleOutOfRange),
            "a backoff can never raise"
        );
        assert_eq!(
            a.arm(TrialDirectives {
                sites: TrialSite::WaveWidth.bit(),
                ..TrialDirectives::NONE
            }),
            Err(TrialArmRefusal::SiteWithoutRule)
        );
        assert!(!a.is_armed());
        a.arm(TrialDirectives {
            width: Some(WidthRule::SHIPPED),
            sites: TrialSite::WaveWidth.bit(),
            ..TrialDirectives::NONE
        })
        .expect("positive control");
        assert_eq!(a.width(TrialSite::WaveWidth), Some(WidthRule::SHIPPED));
        assert_eq!(
            a.width(TrialSite::LowdimWidth),
            None,
            "only armed sites answer"
        );
        assert_eq!(a.width(TrialSite::AlphaLr), None, "not a width site");
    }

    #[test]
    fn router_r6_fault_disarms_the_trial_authority() {
        let _serial = globals();
        guard::reset();
        let trials = crate::trial_authority::global();
        trials
            .arm(TrialDirectives {
                width: Some(WidthRule::SHIPPED),
                lr_backoff: Some(LrRule::SHIPPED),
                split_restart: Some(RestartRule::SHIPPED),
                attack_halving: Some(HalvingRule::SHIPPED),
                sites: TrialSite::ALL.iter().fold(0, |b, s| b | s.bit()),
            })
            .expect("arm");
        assert!(trials.lr_backoff().is_some(), "positive control");
        let fault = guard::fail_open("injected-r6", || -> u32 {
            panic!("router-r6 injected panic")
        })
        .expect_err("the injected panic is contained");
        assert!(!fault.skipped);
        assert!(
            !trials.is_armed(),
            "a contained fault disarms the trial authority"
        );
        for site in TrialSite::ALL {
            assert!(!trials.site_armed(site));
        }
        assert_eq!(trials.attack_halving(), None);
        guard::reset();
        super::bus::reset();
    }

    /// The trial budget is per ROW: a lane run more than once (per clause,
    /// per ascent) starts its machine with the trials the row already spent,
    /// counted by the authority even past its event cap.
    #[test]
    fn router_r6_trial_budget_is_per_row() {
        let a = TrialAuthority::new();
        a.arm(TrialDirectives {
            width: Some(WidthRule::SHIPPED),
            lr_backoff: Some(LrRule::SHIPPED),
            sites: TrialSite::LowdimWidth.bit() | TrialSite::AlphaLr.bit(),
            ..TrialDirectives::NONE
        })
        .expect("arm");
        let started = |site: TrialSite| TrialEvent {
            rule: site.rule().to_owned(),
            site: site.token().to_owned(),
            knob: site.knob().to_owned(),
            dir: TrialDir::Up,
            decision: TrialDecision::Started,
            from: 1.0,
            to: 2.0,
            metric: 0.0,
            baseline: 0.0,
            unit: 0,
            window: 0,
            why: "rise".to_owned(),
        };
        // Fill the stored ledger first: the budget is counted past the cap.
        for _ in 0..crate::trial_authority::MAX_EVENTS {
            a.record(&TrialEvent {
                decision: TrialDecision::Blocked,
                ..started(TrialSite::WaveWidth)
            });
        }
        a.record(&started(TrialSite::LowdimWidth));
        assert_eq!(a.started(TrialSite::LowdimWidth), 1);
        assert_eq!(a.started(TrialSite::WaveWidth), 0, "blocked is not started");
        assert_eq!(a.started(TrialSite::AlphaLr), 0, "per site");
        // A second lane run gets ONE trial left, not two.
        let (m, mut st) = primed(256);
        let mut m = m.with_prior_trials(a.started(TrialSite::LowdimWidth));
        let mut rate = 1_600.0;
        let mut started_now = 0;
        for _ in 0..20 {
            rate *= 1.2;
            started_now += window(&mut m, &mut st, rate, Some(0.1))
                .iter()
                .filter(|e| e.decision == TrialDecision::Started)
                .count();
        }
        assert_eq!(
            started_now, 1,
            "the row's budget of two is shared across runs"
        );
        let backoff = lr().with_prior_trials(2);
        assert_eq!(backoff.trials_used(), 2);
        // Re-arming starts a new row.
        a.arm(TrialDirectives {
            width: Some(WidthRule::SHIPPED),
            sites: TrialSite::LowdimWidth.bit(),
            ..TrialDirectives::NONE
        })
        .expect("re-arm");
        assert_eq!(a.started(TrialSite::LowdimWidth), 0);
    }

    #[test]
    fn router_r6_trial_snapshot_round_trips_through_json() {
        let a = TrialAuthority::new();
        a.arm(TrialDirectives {
            width: Some(WidthRule::SHIPPED),
            sites: TrialSite::LowdimWidth.bit(),
            ..TrialDirectives::NONE
        })
        .expect("arm");
        a.record(&TrialEvent {
            rule: "rule.trial_width".to_owned(),
            site: "lowdim".to_owned(),
            knob: "lowdim.batch".to_owned(),
            dir: TrialDir::Up,
            decision: TrialDecision::Started,
            from: f64::NAN,
            to: f64::INFINITY,
            metric: f64::NEG_INFINITY,
            baseline: 1.5,
            unit: 7,
            window: 2,
            why: "rise".to_owned(),
        });
        let snap = a.snapshot();
        let text = serde_json::to_string(&snap).expect("serialize");
        let back: crate::trial_authority::TrialSnapshot =
            serde_json::from_str(&text).expect("deserialize");
        assert_eq!(back, snap);
        assert_eq!(snap.counts.started, 1);
        for _ in 0..crate::trial_authority::MAX_EVENTS + 5 {
            a.record(&snap.events[0]);
        }
        assert_eq!(
            a.snapshot().events.len(),
            crate::trial_authority::MAX_EVENTS
        );
        assert_eq!(a.snapshot().dropped, 6);
    }

    #[test]
    fn router_r6_arithmetic_never_panics_on_extremes() {
        let mut rng = Rng(0x5eed_6604);
        for _ in 0..200 {
            let input = random_input(&mut rng);
            let _ = commit(&input, EffectiveMode::Acting, &TrialSwitches::ALL_ON);
            let _ = start_refusal(
                rng.next(),
                [0.0, f64::NAN, f64::INFINITY, 1e-300][rng.below(4) as usize],
                [f64::NAN, -1.0, 0.0, f64::MAX][rng.below(4) as usize],
                [None, Some(f64::NAN), Some(f64::INFINITY)][rng.below(3) as usize],
                u32::MAX,
                u32::MAX,
                f64::NAN,
                f64::NAN,
            );
            let base = rng.next().max(1);
            if let Some(range) = WidthRange::around(base, 4, 1, u64::MAX) {
                let mut m = AimdTrial::new(WidthRule::SHIPPED, "r", "s", "k", base, range)
                    .expect("machine");
                for _ in 0..50 {
                    let _ = m.observe(WidthObs {
                        units: rng.next(),
                        closes: rng.next(),
                        t_s: f64::from_bits(rng.next()),
                        remaining_s: Some(f64::from_bits(rng.next())),
                        lease_use: Some(f64::from_bits(rng.next())),
                    });
                    assert!(range.contains(m.value()));
                }
            }
            let mut h = halving();
            for _ in 0..30 {
                let (arm, _) = h.next_arm();
                let _ = h.done_restart(
                    arm,
                    f64::from_bits(rng.next()),
                    f64::from_bits(rng.next()),
                    f64::from_bits(rng.next()),
                );
            }
            let _ = h.done_restart(usize::MAX, 1.0, 1.0, 0.0);
            let mut s = split();
            for _ in 0..30 {
                let _ = s.observe(SplitObs {
                    units: rng.next(),
                    closed_volume: f64::from_bits(rng.next()),
                    open: rng.next(),
                    children: rng.next(),
                    zero_lift: rng.next(),
                    t_s: f64::from_bits(rng.next()),
                    remaining_s: Some(f64::from_bits(rng.next())),
                });
            }
        }
        let _ = Duration::ZERO;
    }
}

// ============================================================================
// #admission-precision — the specification's routing shape, the lanes' own
// route checks, and each routable consumer's site.
// ============================================================================

mod admission_precision {
    use super::super::admission::{
        self, dag_possible, disjunctive_lane, graph_multi_objective, graph_network,
        graph_network_possible, huge_boxed_clause_disjunction, reaches_disjunctive_lane,
        relational_verifier, sequential_conjunction_upgrade, site_reachable, GraphRouteTerms,
        MoAggregation, HUGE_BOXED_CLAUSE_DISJUNCTION_MIN,
    };
    use super::super::decl::{
        Admission, AdmitSite, AdmitWhen, OpKind, OpSet, SpecArchetype, SpecSet, DECLINE_SPEC,
    };
    use super::super::facts::InstanceFacts;
    use super::super::spec::SpecShape;
    use super::{host, Rng};

    fn shape(atoms: u64, clauses: u64) -> SpecShape {
        SpecShape {
            atoms,
            clauses,
            max_clause_atoms: if clauses == 0 {
                atoms
            } else {
                atoms.div_ceil(clauses)
            },
            ..SpecShape::default()
        }
    }

    /// A random shape over every flag and the counts around each threshold.
    fn random_shape(rng: &mut Rng) -> SpecShape {
        let atoms = rng.below(5);
        let clauses = rng.below(4);
        SpecShape {
            atoms,
            clauses,
            max_clause_atoms: rng.below(4),
            boxed_clauses: [0, 1, 511, 512, 513, 60_000][rng.below(6) as usize],
            disjunction: rng.coin(),
            relational_atoms: rng.coin(),
            argmax_margin: rng.coin(),
            multi_clause_disjunction: rng.coin(),
            boxed_clause_disjunction: rng.coin(),
            same_lhs_reducible: rng.coin(),
            inputs: rng.below(100),
            perturbed_inputs: rng.below(100),
            outputs: rng.below(12),
        }
    }

    fn random_ops(rng: &mut Rng) -> OpSet {
        let mut ops = OpSet::EMPTY;
        for kind in OpKind::ALL {
            if rng.below(4) == 0 {
                ops = ops.with(kind);
            }
        }
        ops
    }

    /// The archetype mapping is the seal pipeline's offline rule: argmax
    /// robustness first, then one (effective) clause, else a disjunction.
    #[test]
    fn admission_precision_archetype_mapping() {
        let robust = SpecShape {
            argmax_margin: true,
            ..shape(9, 9)
        };
        assert_eq!(robust.archetype(), SpecArchetype::Robustness);
        // A two-class robustness property is ONE single-atom clause: still
        // robustness (checked first), as the offline classifier says.
        let two_class = SpecShape {
            argmax_margin: true,
            ..shape(1, 1)
        };
        assert_eq!(two_class.archetype(), SpecArchetype::Robustness);
        assert_eq!(shape(1, 1).archetype(), SpecArchetype::Halfspace);
        assert_eq!(shape(4, 1).archetype(), SpecArchetype::Halfspace);
        assert_eq!(
            shape(4, 0).archetype(),
            SpecArchetype::Halfspace,
            "ungrouped atoms are one conjunctive clause"
        );
        assert_eq!(shape(2, 2).archetype(), SpecArchetype::Disjunctive);
        assert_eq!(shape(24, 2).archetype(), SpecArchetype::Disjunctive);
        assert_eq!(shape(0, 0).effective_clauses(), 0);
        for spec in [robust, two_class, shape(1, 1), shape(2, 2)] {
            assert!(SpecSet::AFFINE.contains(spec.archetype()));
        }
        assert!(!SpecSet::AFFINE.contains(SpecArchetype::Relational));
        assert!(!SpecSet::AFFINE.contains(SpecArchetype::Nonlinear));
        assert_eq!(shape(1, 1).flag_tokens(), "-");
        assert_eq!(
            SpecShape {
                disjunction: true,
                multi_clause_disjunction: true,
                relational_atoms: true,
                argmax_margin: true,
                ..shape(9, 9)
            }
            .flag_tokens(),
            "or+multi+rel+argmax"
        );
    }

    /// The lane checks, each against the lane's OWN inline expression as it
    /// stood before #admission-precision (verbatim oracles over the same
    /// inputs), on every combination of the bits they read.
    #[test]
    fn admission_precision_lane_checks_equal_the_inline_expressions() {
        for bits in 0..(1u32 << 6) {
            for atoms in 0..4u64 {
                let bit = |i: u32| bits >> i & 1 == 1;
                let spec = SpecShape {
                    atoms,
                    relational_atoms: bit(0),
                    boxed_clause_disjunction: bit(1),
                    multi_clause_disjunction: bit(2),
                    disjunction: bit(3),
                    same_lhs_reducible: bit(4),
                    ..SpecShape::default()
                };
                let use_relu_split = bit(5);
                // dispatch.rs: `output_constraints.len() > 1 || has_relational
                //   || has_boxed_clause_disjunction()`.
                assert_eq!(
                    relational_verifier(&spec),
                    atoms > 1 || spec.relational_atoms || spec.boxed_clause_disjunction
                );
                // verify/mod.rs: `has_multi_constraint_disjunction() ||
                //   has_boxed_clause_disjunction()`.
                assert_eq!(
                    disjunctive_lane(&spec),
                    spec.multi_clause_disjunction || spec.boxed_clause_disjunction
                );
                // verify/graph.rs: `total_constraint_count >= 2 &&
                //   (use_relu_split || !is_disjunction)`, `conjunctive =
                //   !is_disjunction`.
                let old_mo = atoms >= 2 && (use_relu_split || !spec.disjunction);
                let new_mo = graph_multi_objective(&spec)
                    .filter(|agg| use_relu_split || *agg == MoAggregation::Conjunctive);
                assert_eq!(new_mo.is_some(), old_mo);
                if let Some(agg) = new_mo {
                    assert_eq!(agg == MoAggregation::Conjunctive, !spec.disjunction);
                }
                // verify/mod.rs should_upgrade_sequential_conjunction_to_graph:
                //   `aggregation != Conjunctive || len < 2 => false;
                //    use_relu_split || reduction.is_none()`.
                let old_upgrade = !(spec.disjunction || atoms < 2)
                    && (use_relu_split || !spec.same_lhs_reducible);
                assert_eq!(
                    sequential_conjunction_upgrade(&spec, use_relu_split),
                    old_upgrade
                );
            }
        }
        // model_load.rs: `... .count() >= 512` and
        //   `needs_graph || use_graph_for_conv || huge`.
        assert_eq!(HUGE_BOXED_CLAUSE_DISJUNCTION_MIN, 512);
        for boxed in [0, 1, 511, 512, 513, 120_000] {
            assert_eq!(huge_boxed_clause_disjunction(boxed), boxed >= 512);
        }
        for bits in 0..8u32 {
            let terms = GraphRouteTerms {
                n_ary: bits & 1 == 1,
                conv: bits & 2 == 2,
                huge_boxed_disjunction: bits & 4 == 4,
            };
            assert_eq!(
                graph_network(terms),
                terms.n_ary || terms.conv || terms.huge_boxed_disjunction
            );
        }
    }

    /// A literal model of the lanes' dispatch with EVERY policy choice the
    /// sites omit enumerated: whether the consumer at `site` is reached on a
    /// row (`ops` are the loader's real kinds).
    fn reached(site: AdmitSite, ops: OpSet, spec: &SpecShape, policy: u32) -> bool {
        let p = |i: u32| policy >> i & 1 == 1;
        let (relu, conv_to_graph, f64_cell, vgg_rewrite, softmax_rewrite, per_disjunct) =
            (p(0), p(1), p(2), p(3), p(4), p(5));
        let has = |kind: OpKind| ops.contains(kind);
        // model_load.rs: the graph the loader builds and its decision.
        let needs_graph = has(OpKind::Residual)
            || has(OpKind::Concat)
            || has(OpKind::Bilinear)
            || has(OpKind::Norm)
            || has(OpKind::Other)
            || (vgg_rewrite && has(OpKind::Pool));
        let conv2d = has(OpKind::Conv) || has(OpKind::ConvTranspose);
        let conv_graph = conv2d && conv_to_graph;
        let huge = !needs_graph && !conv_graph && spec.boxed_clauses >= 512 && f64_cell;
        let is_graph = needs_graph || conv_graph || huge;
        let dag = needs_graph || (is_graph && softmax_rewrite && has(OpKind::Softmax));
        // dispatch.rs / verify/mod.rs.
        let relational = spec.atoms > 1 || spec.relational_atoms || spec.boxed_clause_disjunction;
        let disj_lane =
            relational && (spec.multi_clause_disjunction || spec.boxed_clause_disjunction);
        let impl_route = relational && !disj_lane;
        match site {
            // The screen: inside the disjunctive lane on a graph, unless an
            // earlier structural exit took the row (each needs ReLU split).
            AdmitSite::DisjunctiveLaneGraph => disj_lane && is_graph && !relu,
            AdmitSite::DisjunctiveMultiObjective => {
                relu && ((disj_lane && per_disjunct)
                    || (disj_lane && conv_graph && spec.disjunction && spec.atoms >= 2)
                    || (impl_route && is_graph && spec.disjunction && spec.atoms >= 2))
            }
            AdmitSite::GraphInputSplit => {
                !relu
                    && (is_graph
                        || (impl_route
                            && !spec.disjunction
                            && spec.atoms >= 2
                            && !spec.same_lhs_reducible)
                        || disj_lane)
            }
            AdmitSite::DagGraph => relu && dag,
            // #kfsb-production: the graph kFSB selector, inside a graph
            // ReLU-split BaB — the DomainList lane on a graph model, the
            // ReLU-split conjunction upgrade of a sequential model, or the
            // disjunctive lane's per-disjunct graph BaB.
            AdmitSite::GraphReluSplit => {
                let relu_node =
                    has(OpKind::Relu) || (vgg_rewrite && has(OpKind::Pool)) || has(OpKind::Other);
                relu && relu_node
                    && (is_graph
                        || (impl_route && !spec.disjunction && spec.atoms >= 2)
                        || disj_lane)
            }
        }
    }

    /// NECESSARY AND EXACT: with a known shape, a site is reachable iff SOME
    /// policy the lanes read reaches its consumer in the literal dispatch
    /// model above — so a decline is never a row any configuration would
    /// engage on, and an admission always names one that could.
    #[test]
    fn admission_precision_sites_are_the_policy_closure_of_the_dispatch() {
        let mut rng = Rng(0x5eed_a001);
        let mut seen = [[0usize; 2]; AdmitSite::ALL.len()];
        for _ in 0..20_000 {
            let ops = random_ops(&mut rng);
            let spec = random_shape(&mut rng);
            for (i, site) in AdmitSite::ALL.into_iter().enumerate() {
                let any_policy = (0..64).any(|policy| reached(site, ops, &spec, policy));
                let decided = site_reachable(site, ops, Some(&spec));
                assert_eq!(
                    decided,
                    Some(any_policy),
                    "{site:?} on ops {ops:?} spec {spec:?}"
                );
                seen[i][usize::from(any_policy)] += 1;
            }
        }
        for (i, [declined, admitted]) in seen.into_iter().enumerate() {
            assert!(
                declined > 0 && admitted > 0,
                "positive control: {:?} both declines and admits",
                AdmitSite::ALL[i]
            );
        }
    }

    /// An unknown shape is deferred unless the operators alone decide, and a
    /// known shape never contradicts a deferral: every decline at start is
    /// also a decline once the shape is known, whatever it is.
    #[test]
    fn admission_precision_unknown_shapes_defer() {
        let mut rng = Rng(0x5eed_a002);
        for _ in 0..5_000 {
            let ops = random_ops(&mut rng);
            let spec = random_shape(&mut rng);
            for site in AdmitSite::ALL {
                match site_reachable(site, ops, None) {
                    None => {}
                    Some(at_start) => assert_eq!(
                        site_reachable(site, ops, Some(&spec)),
                        Some(at_start),
                        "{site:?}: an operator-decided answer is the shape-known answer"
                    ),
                }
            }
        }
        let chain = OpSet::EMPTY.with(OpKind::Linear).with(OpKind::Relu);
        assert_eq!(
            site_reachable(AdmitSite::DagGraph, chain, None),
            Some(false),
            "a chain signature decides the DAG site without the shape"
        );
        assert_eq!(
            site_reachable(AdmitSite::DisjunctiveLaneGraph, chain, None),
            None
        );
        let resnet = chain.with(OpKind::Conv).with(OpKind::Residual);
        assert_eq!(
            site_reachable(AdmitSite::GraphInputSplit, resnet, None),
            Some(true)
        );
        assert_eq!(dag_possible(chain.with(OpKind::Softmax), None), None);
        assert_eq!(
            graph_network_possible(chain, Some(&shape(1, 1))),
            Some(false)
        );
    }

    /// PRESENCE ONLY: adding operator kinds never turns an admission into a
    /// decline (the scan's signature is a superset of the loader's kinds, so a
    /// spurious kind can only widen).
    #[test]
    fn admission_precision_sites_are_monotone_in_the_operators() {
        let mut rng = Rng(0x5eed_a003);
        for _ in 0..10_000 {
            let ops = random_ops(&mut rng);
            let more = OpSet(ops.0 | random_ops(&mut rng).0);
            let spec = random_shape(&mut rng);
            for site in AdmitSite::ALL {
                for known in [None, Some(&spec)] {
                    if site_reachable(site, ops, known) != Some(false) {
                        assert_ne!(
                            site_reachable(site, more, known),
                            Some(false),
                            "{site:?}: {ops:?} admitted but the wider {more:?} declined"
                        );
                    }
                }
            }
        }
    }

    /// The precondition: a site decline is `site-unreached:<site>`, checked
    /// after the archetype; an unknown shape is no refusal.
    #[test]
    fn admission_precision_evaluate_declines_and_defers() {
        let h = host();
        let mut facts = InstanceFacts::known_empty();
        facts.ops = OpSet::EMPTY.with(OpKind::Linear).with(OpKind::Relu);
        facts.ops_complete = true;
        let screen = AdmitWhen {
            specs: SpecSet::AFFINE,
            site: Some(AdmitSite::DisjunctiveLaneGraph),
            ..AdmitWhen::ANY
        };
        assert_eq!(
            screen.evaluate(&facts, &h),
            Admission::Admit,
            "no shape at start: deferred"
        );
        let halfspace = shape(1, 1);
        facts.spec = Some(halfspace.archetype());
        facts.spec_shape = Some(halfspace);
        assert_eq!(
            screen.evaluate(&facts, &h),
            Admission::Decline("site-unreached:disjunctive_lane_graph")
        );
        // A boxed single clause enters the disjunctive lane; on an MLP the
        // screen still needs the graph: 512 boxed clauses route it there.
        let boxed = SpecShape {
            boxed_clauses: 512,
            boxed_clause_disjunction: true,
            ..shape(1, 1)
        };
        facts.spec = Some(boxed.archetype());
        facts.spec_shape = Some(boxed);
        assert_eq!(screen.evaluate(&facts, &h), Admission::Admit);
        facts.spec_shape = Some(SpecShape {
            boxed_clauses: 511,
            ..boxed
        });
        assert_eq!(
            screen.evaluate(&facts, &h),
            Admission::Decline("site-unreached:disjunctive_lane_graph")
        );
        // The archetype check runs first.
        facts.spec = Some(SpecArchetype::Nonlinear);
        facts.spec_shape = None;
        assert_eq!(
            screen.evaluate(&facts, &h),
            Admission::Decline(DECLINE_SPEC)
        );
        for site in AdmitSite::ALL {
            assert!(site.decline().ends_with(site.token()));
        }
        assert_eq!(
            admission::N_ARY_AFTER_LOAD,
            OpSet::EMPTY.with(OpKind::Softmax)
        );
    }

    /// The robustness rows the router routes margin-row consumers on read as
    /// robustness to the disjunctive checks too: a robustness property is a
    /// multi-clause disjunction of relational atoms (it reaches the
    /// disjunctive lane and the non-conjunctive multi-objective BaB).
    #[test]
    fn admission_precision_robustness_reaches_the_disjunctive_sites() {
        let robust = SpecShape {
            disjunction: true,
            relational_atoms: true,
            argmax_margin: true,
            multi_clause_disjunction: true,
            ..shape(9, 9)
        };
        assert!(reaches_disjunctive_lane(&robust));
        assert_eq!(
            graph_multi_objective(&robust),
            Some(MoAggregation::Disjunctive)
        );
        let vit = OpSet::EMPTY
            .with(OpKind::Linear)
            .with(OpKind::Conv)
            .with(OpKind::Bilinear)
            .with(OpKind::Softmax)
            .with(OpKind::Norm)
            .with(OpKind::Residual);
        for site in AdmitSite::ALL {
            // #kfsb-production: the graph ReLU-split site needs a ReLU-capable
            // operator, which this signature lacks; with one it is reached.
            let ops = if site == AdmitSite::GraphReluSplit {
                assert_eq!(
                    site_reachable(site, vit, Some(&robust)),
                    Some(false),
                    "no ReLU-capable kind: nothing for a ReLU-split selector to split"
                );
                vit.with(OpKind::Relu)
            } else {
                vit
            };
            assert_eq!(
                site_reachable(site, ops, Some(&robust)),
                Some(true),
                "{site:?}"
            );
        }
        // A single relational atom over an MLP (the 2-class NLP shape) reaches
        // none of the graph sites.
        let two_class = SpecShape {
            relational_atoms: true,
            argmax_margin: true,
            ..shape(1, 1)
        };
        let mlp = OpSet::EMPTY.with(OpKind::Linear).with(OpKind::Relu);
        for site in AdmitSite::ALL {
            assert_eq!(
                site_reachable(site, mlp, Some(&two_class)),
                Some(false),
                "{site:?}"
            );
        }
    }

    /// A site is a static input: it changes the plan hash.
    #[test]
    fn admission_precision_site_is_hashed() {
        let mut rng = Rng(0x5eed_a004);
        let mut input = super::random_input(&mut rng);
        input.capabilities.push(super::super::plan::CapInput {
            route: Some(super::super::plan::RouteSpec {
                sr: super::super::decl::SrLevel::Decl,
                parent: None,
                admit_when: AdmitWhen::ANY,
                envelope_ops: OpSet::EMPTY,
                envelope_specs: SpecSet::EMPTY,
                risk: super::super::decl::KnobRisk::High,
                settings: Vec::new(),
                slot: None,
            }),
            ..super::cap(
                "s",
                super::CapClass::Rule,
                super::LedgerState::Shadow,
                super::super::decl::ProgressKind::Opaque,
                None,
                super::AdmissionState::Admit,
            )
        });
        let base = super::hash::static_hash(&input);
        let last = input.capabilities.len() - 1;
        for site in AdmitSite::ALL {
            let mut sited = input.clone();
            if let Some(route) = sited.capabilities[last].route.as_mut() {
                route.admit_when.site = Some(site);
            }
            assert_ne!(super::hash::static_hash(&sited), base, "{site:?}");
        }
    }
}
