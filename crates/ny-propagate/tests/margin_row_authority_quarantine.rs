use ny_propagate::margin_row::{
    run_margin_row_lane, run_margin_row_lane_with_head, BabStats, MarginRowOutcome, TwinOpSpec,
    TwinSpec,
};

#[derive(Debug, PartialEq, Eq)]
struct StableStats {
    root_bound_bits: u64,
    tree_classes: Vec<usize>,
    root_closed_classes: usize,
    expansions: usize,
    domains_created: usize,
    closed: usize,
    max_depth: usize,
    mono_raw_dips: usize,
    mono_worst_bits: u64,
    stop: String,
    class_runs: usize,
    epochs_attempted: usize,
    epochs_closed: usize,
    ledger_ok: Option<bool>,
}

impl From<BabStats> for StableStats {
    fn from(stats: BabStats) -> Self {
        Self {
            root_bound_bits: stats.root_bound.to_bits(),
            tree_classes: stats.tree_classes,
            root_closed_classes: stats.root_closed_classes,
            expansions: stats.expansions,
            domains_created: stats.domains_created,
            closed: stats.closed,
            max_depth: stats.max_depth,
            mono_raw_dips: stats.mono_raw_dips,
            mono_worst_bits: stats.mono_worst.to_bits(),
            stop: stats.stop,
            class_runs: stats.class_runs.len(),
            epochs_attempted: stats.epochs_attempted,
            epochs_closed: stats.epochs_closed,
            ledger_ok: stats.ledger_ok,
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
enum StableOutcome {
    Unsat(StableStats),
    Unknown {
        reason: String,
        stats: Option<StableStats>,
    },
}

impl From<MarginRowOutcome> for StableOutcome {
    fn from(outcome: MarginRowOutcome) -> Self {
        match outcome {
            MarginRowOutcome::Unsat(stats) => Self::Unsat(stats.into()),
            MarginRowOutcome::Unknown { reason, stats } => Self::Unknown {
                reason,
                stats: stats.map(Into::into),
            },
        }
    }
}

fn structurally_valid_tiny_spec() -> TwinSpec {
    TwinSpec {
        n_in: 1,
        ops: vec![
            TwinOpSpec::Conv {
                input: 0,
                weight: vec![1.0],
                bias: vec![0.0],
                bias_err: vec![0.0],
                weight_rel_err: 0.0,
                kernel: (1, 1, 1, 1),
                stride: (1, 1),
                pads: (0, 0, 0, 0),
                ishape: (1, 1, 1),
                oshape: (1, 1, 1),
            },
            TwinOpSpec::Relu { input: 1 },
            TwinOpSpec::Flatten { input: 2 },
            TwinOpSpec::Gemm {
                input: 3,
                weight: vec![1.0],
                bias: vec![0.0],
                shape: (1, 1),
            },
            TwinOpSpec::Relu { input: 4 },
            TwinOpSpec::Gemm {
                input: 5,
                weight: vec![1.0, -1.0],
                bias: vec![1.0, 0.0],
                shape: (2, 1),
            },
        ],
    }
}

#[test]
fn production_entrypoint_runs_certified_lane() {
    // Quarantine lifted 2026-07-18: the production entrypoint now runs the certified
    // algorithm. It only ever returns Unsat or Unknown (never Sat), and never the old
    // "authority quarantine" stub.
    assert!(ny_propagate::margin_row::margin_row_bab_enabled());
    let outcome = run_margin_row_lane(
        &structurally_valid_tiny_spec(),
        &[0.0],
        &[1.0],
        0,
        &[1],
        None,
        100,
    );
    match outcome {
        MarginRowOutcome::Unknown { reason, .. } => {
            assert!(
                !reason.contains("authority quarantine"),
                "lane still quarantined: {reason}"
            );
        }
        // A certified `unsat` is a legitimate production outcome now.
        MarginRowOutcome::Unsat(_) => {}
    }
}

#[test]
fn external_head_wrapper_is_gate_on_semantic_alias() {
    let spec = structurally_valid_tiny_spec();
    let (ordinary, quarantined) =
        ny_test_utils::env::with_serialized_env_vars(&[("NY_MARGIN_ROW_HEAD_INJECT", "1")], || {
            let ordinary = run_margin_row_lane(&spec, &[0.0], &[1.0], 0, &[1], None, 100);
            let quarantined = run_margin_row_lane_with_head(
                &spec,
                &[0.0],
                &[1.0],
                0,
                &[1],
                None,
                100,
                // Deliberately malformed and non-finite. A hard alias must
                // neither inspect this payload nor let the old environment
                // spelling activate any `initial_ybox` authority.
                Some((vec![f64::NAN, f64::INFINITY], vec![])),
            );
            (ordinary, quarantined)
        });

    assert_eq!(
        StableOutcome::from(ordinary),
        StableOutcome::from(quarantined)
    );
}

/// A structurally INVALID spec (head is not `Gemm -> Relu -> Gemm`): the
/// production entrypoint must fail closed to `Unknown`, never decide.
fn structurally_invalid_spec() -> TwinSpec {
    TwinSpec {
        n_in: 1,
        ops: vec![TwinOpSpec::Relu { input: 0 }],
    }
}

/// THE MOAT, at the production entrypoint: a structural mismatch fails
/// closed to `Unknown`. The lane's outcome type admits no `Sat`, so this is
/// the only way it can be wrong — by deciding something it cannot model.
#[test]
fn production_entrypoint_fails_closed_on_structural_mismatch() {
    match run_margin_row_lane(
        &structurally_invalid_spec(),
        &[0.0],
        &[1.0],
        0,
        &[1],
        None,
        100,
    ) {
        MarginRowOutcome::Unknown { .. } => {}
        MarginRowOutcome::Unsat(_) => {
            panic!("MOAT VIOLATION: decided an instance outside the supported class")
        }
    }
}
