// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #router-r7 — the OFFLINE PRIORS v1 gates (ROUTER_PROGRAM §5.5, §10 R7).
//!
//! The compiled cost-model table (`src/router/priors_v1.tsv`) is produced by
//! `scripts/router_priors/build_priors.py fit` from a committed training set;
//! the pipeline also writes the ANONYMISED unit manifest these tests read
//! (`tests/data/priors_v1_rows.tsv`: opaque family ids, hashed model ids). This
//! file is the SECOND implementation that checks the first:
//!
//! * the table loads, validates, and carries its own digest in its version;
//! * the reader refuses a table that breaks any gate (each mutation names its
//!   gate, so a weakened check fails here);
//! * every shipped coefficient re-derives from the manifest (the table is
//!   what the pipeline says it fit);
//! * leave-one-family-out holds each family out ENTIRELY, and its recorded
//!   overfit gap and win/loss count re-derive;
//! * a classifier trained on the prior's INPUT KEY cannot recover the family
//!   on held-out models beyond the declared bound — with a POSITIVE CONTROL
//!   showing the same classifier on the raw size DOES (so the test has power:
//!   size fingerprints, the key does not);
//! * every shipped cell spans two families within the purity bound;
//! * the prior names no category and no path, and its outputs are keyed only
//!   by the closed operator-key vocabulary (the name-blind source ratchet);
//! * the plan reads the prior only when it is present: without it the plan
//!   and the static hash are R5's, bit for bit; with it a deferred lane's
//!   priced rung is contracted and an unfit rung leaves the slot on today's
//!   deadline.
//!
//! Review additions (#router-r7): the bound above covers the KEY only — the
//! prior's logged OUTPUT does identify the family (a monotone map of the
//! size), so the guard on a size-bearing prior is its capacity, pinned here;
//! a per-clause phase scales by the row's clauses and fails closed; every
//! field of the prior's view moves the static hash; and a prior contracts
//! only the PRICED deferred lanes.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use ny_core::deadline_authority::LaneSlot;
use ny_core::router::acting::{self, RuleSwitches};
use ny_core::router::decl::{
    AmendmentSet, LaneRole, OpKind, OpSet, ProgressKind, SafePoints, WorkUnit,
};
use ny_core::router::facts::{DeviceQual, HostClass, HostEnvelope, InstanceFacts};
use ny_core::router::hash;
use ny_core::router::plan::{
    self, AdmissionState, BudgetView, CapClass, CapInput, LedgerState, PolicyInput,
};
use ny_core::router::priors::{
    self, CostKey, LayerB, PhasePrediction, PricedCap, PriorError, PriorTable, PriorView,
    UnpricedCap, COST_TABLE_V1, FINGERPRINT_EXCESS_MAX, MIN_FAMILIES_PER_CELL,
    OVERFIT_GAP_MAX_LOG2, PURITY_MAX, ROOT_CELL, RULE_PRIORS, SLOPE_MAX, SLOPE_MIN,
};
use ny_core::router::{EffectiveMode, PRIORS_VERSION, ROUTER_SCHEMA};

fn crate_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn read(rel: &str) -> String {
    let path = crate_root().join(rel);
    let text =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    assert!(
        text.len() > 200,
        "{} is suspiciously short ({} bytes): an empty file scans clean",
        path.display(),
        text.len()
    );
    text
}

fn table() -> PriorTable {
    PriorTable::compiled().unwrap_or_else(|e| panic!("the compiled prior must load: {e}"))
}

// ---- the anonymised unit manifest -------------------------------------------------------------

#[derive(Debug, Clone)]
struct Unit {
    family: String,
    model: String,
    cell: String,
    x: f64,
    y: f64,
    weight: f64,
}

/// phase -> units, in the manifest's (= the pipeline's) order.
fn manifest() -> BTreeMap<String, Vec<Unit>> {
    let text = read("tests/data/priors_v1_rows.tsv");
    let mut out: BTreeMap<String, Vec<Unit>> = BTreeMap::new();
    for line in text.lines() {
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        assert_eq!(f.len(), 7, "manifest line {line:?}");
        out.entry(f[0].to_owned()).or_default().push(Unit {
            family: f[1].to_owned(),
            model: f[2].to_owned(),
            cell: f[3].to_owned(),
            x: f[4].parse().expect("log2 work"),
            y: f[5].parse().expect("log2 value"),
            weight: 0.0,
        });
    }
    for units in out.values_mut() {
        reweight(units);
    }
    assert!(
        out.values().map(Vec::len).sum::<usize>() > 100,
        "the manifest must carry the training units"
    );
    out
}

/// Family-balanced weights: every family carries total weight 1.
fn reweight(units: &mut [Unit]) {
    let mut per: BTreeMap<String, f64> = BTreeMap::new();
    for u in units.iter() {
        *per.entry(u.family.clone()).or_default() += 1.0;
    }
    for u in units.iter_mut() {
        u.weight = 1.0 / per[&u.family];
    }
}

fn wmedian(mut pairs: Vec<(f64, f64)>) -> f64 {
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
    let total: f64 = pairs.iter().map(|p| p.1).sum();
    let mut acc = 0.0;
    for (value, weight) in &pairs {
        acc += weight;
        if acc >= 0.5 * total {
            return *value;
        }
    }
    pairs.last().expect("non-empty").0
}

fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(f64::total_cmp);
    let n = xs.len();
    assert!(n > 0, "median of nothing");
    if n % 2 == 1 {
        xs[n / 2]
    } else {
        f64::midpoint(xs[n / 2 - 1], xs[n / 2])
    }
}

struct Census {
    families: usize,
    share_max: f64,
    units: usize,
}

fn census(units: &[Unit]) -> BTreeMap<String, Census> {
    let mut fam_w: BTreeMap<String, BTreeMap<String, f64>> = BTreeMap::new();
    let mut weight: BTreeMap<String, f64> = BTreeMap::new();
    let mut count: BTreeMap<String, usize> = BTreeMap::new();
    for u in units {
        *fam_w
            .entry(u.cell.clone())
            .or_default()
            .entry(u.family.clone())
            .or_default() += u.weight;
        *weight.entry(u.cell.clone()).or_default() += u.weight;
        *count.entry(u.cell.clone()).or_default() += 1;
    }
    fam_w
        .into_iter()
        .map(|(cell, fams)| {
            let w = weight[&cell];
            let top = fams.values().copied().fold(0.0_f64, f64::max);
            let c = Census {
                families: fams.len(),
                share_max: if w > 0.0 { top / w } else { 1.0 },
                units: count[&cell],
            };
            (cell, c)
        })
        .collect()
}

fn passing(units: &[Unit]) -> BTreeSet<String> {
    census(units)
        .into_iter()
        .filter(|(_, c)| c.families >= MIN_FAMILIES_PER_CELL as usize && c.share_max <= PURITY_MAX)
        .map(|(cell, _)| cell)
        .collect()
}

struct Fit {
    slope: f64,
    intercepts: BTreeMap<String, f64>,
    spreads: BTreeMap<String, f64>,
}

impl Fit {
    fn predict(&self, cell: &str, x: f64) -> f64 {
        let a = self
            .intercepts
            .get(cell)
            .copied()
            .unwrap_or(self.intercepts[ROOT_CELL]);
        a + self.slope * x
    }
}

const MIN_UNITS_PER_PHASE: usize = 4;

/// The pipeline's `fit_phase`, re-implemented: WLS shared slope, weighted
/// median intercepts for the root and every passing cell, floored spreads.
fn fit(units: &[Unit], spread_floor: f64) -> Option<Fit> {
    if units.len() < MIN_UNITS_PER_PHASE {
        return None;
    }
    let fams: BTreeSet<&str> = units.iter().map(|u| u.family.as_str()).collect();
    if fams.len() < MIN_FAMILIES_PER_CELL as usize {
        return None;
    }
    let sw: f64 = units.iter().map(|u| u.weight).sum();
    let mx = units.iter().map(|u| u.weight * u.x).sum::<f64>() / sw;
    let my = units.iter().map(|u| u.weight * u.y).sum::<f64>() / sw;
    let sxx: f64 = units.iter().map(|u| u.weight * (u.x - mx).powi(2)).sum();
    let sxy: f64 = units
        .iter()
        .map(|u| u.weight * (u.x - mx) * (u.y - my))
        .sum();
    let slope = if sxx > 1e-12 { sxy / sxx } else { 0.0 };
    if !(SLOPE_MIN..=SLOPE_MAX).contains(&slope) {
        return None;
    }
    let mut intercepts = BTreeMap::new();
    intercepts.insert(
        ROOT_CELL.to_owned(),
        wmedian(
            units
                .iter()
                .map(|u| (u.y - slope * u.x, u.weight))
                .collect(),
        ),
    );
    for cell in passing(units) {
        intercepts.insert(
            cell.clone(),
            wmedian(
                units
                    .iter()
                    .filter(|u| u.cell == cell)
                    .map(|u| (u.y - slope * u.x, u.weight))
                    .collect(),
            ),
        );
    }
    let mut spreads = BTreeMap::new();
    for (cell, a) in &intercepts {
        let members: Vec<&Unit> = units
            .iter()
            .filter(|u| cell == ROOT_CELL || &u.cell == cell)
            .collect();
        let mad = wmedian(
            members
                .iter()
                .map(|u| ((u.y - (a + slope * u.x)).abs(), u.weight))
                .collect(),
        );
        spreads.insert(cell.clone(), mad.max(spread_floor));
    }
    Some(Fit {
        slope,
        intercepts,
        spreads,
    })
}

struct Logo {
    families: usize,
    gap_median: f64,
    wins: usize,
    losses: usize,
    splits: Vec<(String, Vec<Unit>, Vec<Unit>)>,
}

fn logo(units: &[Unit], spread_floor: f64) -> Logo {
    let full = fit(units, spread_floor).expect("the full fit exists for a shipped phase");
    let fams: BTreeSet<String> = units.iter().map(|u| u.family.clone()).collect();
    let mut gaps = Vec::new();
    let (mut wins, mut losses) = (0, 0);
    let mut splits = Vec::new();
    for g in &fams {
        let mut train: Vec<Unit> = units.iter().filter(|u| &u.family != g).cloned().collect();
        let test: Vec<Unit> = units.iter().filter(|u| &u.family == g).cloned().collect();
        reweight(&mut train);
        splits.push((g.clone(), train.clone(), test.clone()));
        let Some(held) = fit(&train, spread_floor) else {
            continue;
        };
        let null = wmedian(train.iter().map(|u| (u.y, u.weight)).collect());
        let heldout = median(
            test.iter()
                .map(|u| (u.y - held.predict(&u.cell, u.x)).abs())
                .collect(),
        );
        let fullerr = median(
            test.iter()
                .map(|u| (u.y - full.predict(&u.cell, u.x)).abs())
                .collect(),
        );
        let nullerr = median(test.iter().map(|u| (u.y - null).abs()).collect());
        gaps.push(heldout - fullerr);
        if heldout < nullerr {
            wins += 1;
        } else if heldout > nullerr {
            losses += 1;
        }
    }
    Logo {
        families: gaps.len(),
        gap_median: if gaps.is_empty() {
            f64::INFINITY
        } else {
            median(gaps)
        },
        wins,
        losses,
        splits,
    }
}

/// G2-G6 on `units` (the ship decision a refit makes; no nested G7).
fn gates_pass(units: &[Unit], spread_floor: f64) -> bool {
    let Some(f) = fit(units, spread_floor) else {
        return false;
    };
    let l = logo(units, spread_floor);
    if l.families == 0 || l.gap_median > OVERFIT_GAP_MAX_LOG2 || l.losses > l.wins {
        return false;
    }
    let shipped: BTreeSet<String> = f
        .intercepts
        .keys()
        .filter(|c| *c != ROOT_CELL)
        .cloned()
        .collect();
    let key = |u: &Unit| {
        if shipped.contains(&u.cell) {
            u.cell.clone()
        } else {
            ROOT_CELL.to_owned()
        }
    };
    let (acc, base) = classify(units, &key);
    if acc - base > FINGERPRINT_EXCESS_MAX {
        return false;
    }
    let cen = census(units);
    shipped.iter().all(|cell| cen[cell].families >= 2)
}

/// G7: (leave-one-family-out refits that still pass G2-G6, families).
fn stability(units: &[Unit], spread_floor: f64) -> (usize, usize) {
    let fams: BTreeSet<String> = units.iter().map(|u| u.family.clone()).collect();
    let ok = fams
        .iter()
        .filter(|g| {
            let mut sub: Vec<Unit> = units.iter().filter(|u| &u.family != *g).cloned().collect();
            reweight(&mut sub);
            gates_pass(&sub, spread_floor)
        })
        .count();
    (ok, fams.len())
}

/// The first byte of the hashed model id: the pipeline's fold.
fn fold(model: &str) -> u8 {
    u8::from_str_radix(&model[..2], 16).expect("hex model id") & 1
}

/// G3's classifier: per-key majority family (family-balanced weights),
/// falling back to the global majority for an unseen key; trained on one
/// model-hash fold, scored on the other, both directions; family-balanced
/// accuracy. Returns (accuracy, baseline).
fn classify(units: &[Unit], key: &dyn Fn(&Unit) -> String) -> (f64, f64) {
    let mut accs = Vec::new();
    let mut bases = Vec::new();
    for test_fold in [0_u8, 1] {
        let mut train: Vec<Unit> = units
            .iter()
            .filter(|u| fold(&u.model) != test_fold)
            .cloned()
            .collect();
        reweight(&mut train);
        let test: Vec<&Unit> = units
            .iter()
            .filter(|u| fold(&u.model) == test_fold)
            .collect();
        if train.is_empty() || test.is_empty() {
            continue;
        }
        let mut glob: BTreeMap<String, f64> = BTreeMap::new();
        let mut per_key: BTreeMap<String, BTreeMap<String, f64>> = BTreeMap::new();
        for u in &train {
            *glob.entry(u.family.clone()).or_default() += u.weight;
            *per_key
                .entry(key(u))
                .or_default()
                .entry(u.family.clone())
                .or_default() += u.weight;
        }
        let argmax = |w: &BTreeMap<String, f64>| -> String {
            let mut best: Option<(&String, f64)> = None;
            for (f, v) in w {
                if best.is_none_or(|(_, b)| *v > b) {
                    best = Some((f, *v));
                }
            }
            best.expect("non-empty").0.clone()
        };
        let gmaj = argmax(&glob);
        let kmaj: BTreeMap<String, String> = per_key
            .iter()
            .map(|(k, w)| (k.clone(), argmax(w)))
            .collect();
        let mut hits: Vec<(String, Vec<f64>)> = Vec::new();
        let mut base: Vec<(String, Vec<f64>)> = Vec::new();
        for u in test {
            let pred = kmaj.get(&key(u)).unwrap_or(&gmaj);
            let push = |acc: &mut Vec<(String, Vec<f64>)>, v: f64| match acc
                .iter_mut()
                .find(|(f, _)| f == &u.family)
            {
                Some((_, xs)) => xs.push(v),
                None => acc.push((u.family.clone(), vec![v])),
            };
            push(&mut hits, f64::from(u8::from(pred == &u.family)));
            push(&mut base, f64::from(u8::from(gmaj == u.family)));
        }
        let mean_of = |acc: &[(String, Vec<f64>)]| -> f64 {
            acc.iter()
                .map(|(_, xs)| xs.iter().sum::<f64>() / xs.len() as f64)
                .sum::<f64>()
                / acc.len() as f64
        };
        accs.push(mean_of(&hits));
        bases.push(mean_of(&base));
    }
    assert!(!accs.is_empty(), "a fold is empty");
    (
        accs.iter().sum::<f64>() / accs.len() as f64,
        bases.iter().sum::<f64>() / bases.len() as f64,
    )
}

/// The positive control's classifier: 1-nearest-neighbour on log2(size)
/// (ties to the lower family id), same folds, same family-balanced metric.
fn nearest_size(units: &[Unit]) -> (f64, f64) {
    let mut accs = Vec::new();
    let mut bases = Vec::new();
    for test_fold in [0_u8, 1] {
        let mut train: Vec<Unit> = units
            .iter()
            .filter(|u| fold(&u.model) != test_fold)
            .cloned()
            .collect();
        reweight(&mut train);
        let test: Vec<&Unit> = units
            .iter()
            .filter(|u| fold(&u.model) == test_fold)
            .collect();
        if train.is_empty() || test.is_empty() {
            continue;
        }
        let mut glob: BTreeMap<String, f64> = BTreeMap::new();
        for u in &train {
            *glob.entry(u.family.clone()).or_default() += u.weight;
        }
        let mut gmaj: Option<(&String, f64)> = None;
        for (f, v) in &glob {
            if gmaj.is_none_or(|(_, b)| *v > b) {
                gmaj = Some((f, *v));
            }
        }
        let gmaj = gmaj.expect("non-empty").0.clone();
        let mut hits: BTreeMap<String, Vec<f64>> = BTreeMap::new();
        let mut base: BTreeMap<String, Vec<f64>> = BTreeMap::new();
        for u in test {
            let nn = train
                .iter()
                .min_by(|a, b| {
                    (a.x - u.x)
                        .abs()
                        .total_cmp(&(b.x - u.x).abs())
                        .then(a.family.cmp(&b.family))
                })
                .expect("non-empty training fold");
            hits.entry(u.family.clone())
                .or_default()
                .push(f64::from(u8::from(nn.family == u.family)));
            base.entry(u.family.clone())
                .or_default()
                .push(f64::from(u8::from(gmaj == u.family)));
        }
        let mean_of = |acc: &BTreeMap<String, Vec<f64>>| -> f64 {
            acc.values()
                .map(|xs| xs.iter().sum::<f64>() / xs.len() as f64)
                .sum::<f64>()
                / acc.len() as f64
        };
        accs.push(mean_of(&hits));
        bases.push(mean_of(&base));
    }
    (
        accs.iter().sum::<f64>() / accs.len() as f64,
        bases.iter().sum::<f64>() / bases.len() as f64,
    )
}

fn close(a: f64, b: f64, tol: f64, what: &str) {
    assert!(
        (a - b).abs() <= tol,
        "{what}: re-derived {a} but the table records {b}"
    );
}

// ---- 1. load / version ------------------------------------------------------------------------

#[test]
fn priors_table_loads_validates_and_carries_its_version() {
    let t = table();
    assert_eq!(t.schema, priors::PRIORS_SCHEMA);
    assert_eq!(t.version, "cost-v1");
    assert_eq!(
        t.host,
        HostClass::Gb10,
        "a prior says which host it was measured on"
    );
    assert_eq!(t.trainset_sha256.len(), 64);
    assert!(t.receipts > 100, "receipts={}", t.receipts);
    assert!(t.validate().is_empty(), "{:?}", t.validate());
    let digest: [u8; 32] = {
        use sha2::Digest as _;
        sha2::Sha256::digest(COST_TABLE_V1.as_bytes()).into()
    };
    assert_eq!(t.digest, digest);
    assert_eq!(
        t.version_token(),
        format!("cost-v1+{}", hash::sha8(&digest)),
        "the version token carries the table's own digest"
    );
    assert_ne!(
        t.version_token(),
        PRIORS_VERSION,
        "a read prior never looks uniform"
    );
    assert!(
        matches!(t.layer_b, LayerB::Uniform { .. }),
        "Layer B ships uniform in v1"
    );
    let shipped = t.shipped_phases();
    assert_eq!(shipped, vec!["verifier.setup"], "v1 ships one phase");
    assert!(
        t.refused
            .iter()
            .any(|(p, why)| p == "attack.upfront" && why.contains("G7")),
        "attack.upfront passes G2-G6 on the full data but its pass depends on single families \
         (G7), so it is refused: {:?}",
        t.refused
    );
    assert!(
        t.refused
            .iter()
            .any(|(p, why)| p == "root.bound" && why.contains("G6")),
        "the gates refused root.bound in v1 and the table must say so: {:?}",
        t.refused
    );
    // Parsing is pure and deterministic.
    assert_eq!(PriorTable::parse(COST_TABLE_V1).expect("parses"), t);
}

// ---- 2. the reader refuses a table that breaks a gate -----------------------------------------

fn mutate(from: &str, to: &str) -> String {
    assert!(
        COST_TABLE_V1.contains(from),
        "mutation anchor {from:?} not in the table"
    );
    COST_TABLE_V1.replacen(from, to, 1)
}

fn violations(text: &str) -> Vec<String> {
    match PriorTable::parse(text) {
        Ok(t) => t.validate(),
        Err(PriorError::Parse { what, .. }) => vec![format!("PARSE {what}")],
        Err(e) => vec![e.to_string()],
    }
}

#[test]
fn priors_reader_refuses_a_table_that_breaks_any_gate() {
    let t = table();
    let model = t
        .models
        .iter()
        .find(|m| m.phase == "verifier.setup" && m.cell != ROOT_CELL)
        .expect("a non-root verifier.setup cell");
    let line = |m: &priors::CostModel| {
        format!(
            "model\t{}\t{}\t{}\t{}\t{:.6}\t{:.6}\t{:.6}\t{}\t{}\t{:.6}",
            m.phase,
            m.unit,
            m.work,
            m.cell,
            m.intercept,
            m.slope,
            m.spread,
            m.units,
            m.families,
            m.share_max
        )
    };
    let original = line(model);
    assert!(COST_TABLE_V1.contains(&original), "{original}");
    let with = |edit: &dyn Fn(&mut priors::CostModel)| {
        let mut m = model.clone();
        edit(&mut m);
        mutate(&original, &line(&m))
    };
    let cases: Vec<(&str, String)> = vec![
        ("G5", with(&|m| m.slope += 0.125)),
        ("G1", with(&|m| m.cell = "params>1000000".to_owned())),
        ("G1", with(&|m| m.work = "model_bytes".to_owned())),
        ("G2", with(&|m| m.families = 1)),
        ("G2", with(&|m| m.share_max = 0.9)),
        ("G5", {
            let edit = |m: &mut priors::CostModel| m.slope = 2.5;
            let mut text = COST_TABLE_V1.to_owned();
            for m in t.models.iter().filter(|m| m.phase == "verifier.setup") {
                let mut e = m.clone();
                edit(&mut e);
                text = text.replacen(&line(m), &line(&e), 1);
            }
            text
        }),
        ("G3", {
            let ev = t
                .evidence
                .iter()
                .find(|e| e.phase == "verifier.setup")
                .expect("evidence");
            // Accuracy and excess move TOGETHER (consistent with the baseline),
            // so only the bound itself can refuse it.
            mutate(
                &format!("fp_excess={:.6}", ev.fp_excess),
                &format!("fp_excess={:.6}", FINGERPRINT_EXCESS_MAX + 0.1),
            )
            .replacen(
                &format!("fp_accuracy={:.6}", ev.fp_accuracy),
                &format!(
                    "fp_accuracy={:.6}",
                    ev.fp_baseline + FINGERPRINT_EXCESS_MAX + 0.1
                ),
                1,
            )
        }),
        ("G6", {
            let ev = t
                .evidence
                .iter()
                .find(|e| e.phase == "verifier.setup")
                .expect("evidence");
            mutate(
                &format!("gap_median={:.6}", ev.gap_median),
                &format!("gap_median={:.6}", OVERFIT_GAP_MAX_LOG2 + 0.4),
            )
        }),
        ("G6", {
            let ev = t
                .evidence
                .iter()
                .find(|e| e.phase == "verifier.setup")
                .expect("evidence");
            mutate(
                &format!("wins={}\tlosses={}", ev.wins, ev.losses),
                &format!("wins={}\tlosses={}", ev.losses, ev.wins + 1),
            )
        }),
        ("G4", mutate("isolating=0", "isolating=1")),
        ("G7", {
            let ev = t
                .evidence
                .iter()
                .find(|e| e.phase == "verifier.setup")
                .expect("evidence");
            mutate(
                &format!("stable_folds={}", ev.stable_folds),
                &format!("stable_folds={}", ev.stable_folds - 1),
            )
        }),
        (
            "purity_max",
            mutate("gate\tpurity_max\t0.600000", "gate\tpurity_max\t0.900000"),
        ),
        ("PARSE", mutate("layer_b\tuniform", "layer_b\tfitted")),
        ("PARSE", format!("{COST_TABLE_V1}category\tx\n")),
        ("PARSE", mutate("host\tgb10", "host\tlaptop")),
        (
            "neither shipped nor refused",
            format!("{COST_TABLE_V1}bind\tx\tno.such.phase\n"),
        ),
    ];
    for (gate, text) in cases {
        assert_ne!(
            text, COST_TABLE_V1,
            "{gate}: the mutation must change the table"
        );
        let got = violations(&text);
        assert!(
            got.iter().any(|v| v.contains(gate)),
            "a table breaking {gate} must be refused; got {got:?}"
        );
        if !gate.starts_with("PARSE") {
            assert!(
                PriorTable::parse(&text).is_ok(),
                "{gate}: the mutation must parse so the GATE (not the parser) refuses it"
            );
        }
    }
}

// ---- 3. the table re-derives from its training manifest ---------------------------------------

#[test]
fn priors_table_reproduces_from_its_training_manifest() {
    let t = table();
    let man = manifest();
    for phase in t.shipped_phases() {
        let units = &man[phase];
        let f = fit(units, t.spread_floor_log2).unwrap_or_else(|| panic!("{phase} must re-fit"));
        let models: Vec<_> = t.models.iter().filter(|m| m.phase == phase).collect();
        assert_eq!(
            models.len(),
            f.intercepts.len(),
            "{phase}: shipped cells {:?} vs re-derived {:?}",
            models.iter().map(|m| &m.cell).collect::<Vec<_>>(),
            f.intercepts.keys().collect::<Vec<_>>()
        );
        let cen = census(units);
        for m in models {
            close(f.slope, m.slope, 5e-6, &format!("{phase} slope"));
            close(
                f.intercepts[&m.cell],
                m.intercept,
                5e-6,
                &format!("{phase}[{}] intercept", m.cell),
            );
            close(
                f.spreads[&m.cell],
                m.spread,
                5e-6,
                &format!("{phase}[{}] spread", m.cell),
            );
            if m.cell == ROOT_CELL {
                assert_eq!(m.units as usize, units.len(), "{phase} root units");
            } else {
                let c = &cen[&m.cell];
                assert_eq!(m.units as usize, c.units, "{phase}[{}] units", m.cell);
                assert_eq!(
                    m.families as usize, c.families,
                    "{phase}[{}] families",
                    m.cell
                );
                close(
                    c.share_max,
                    m.share_max,
                    5e-6,
                    &format!("{phase}[{}] share", m.cell),
                );
            }
        }
    }
    // A refused phase is refused by the same arithmetic.
    for (phase, _) in &t.refused {
        let Some(units) = man.get(phase) else {
            continue;
        };
        let (ok, of) = stability(units, t.spread_floor_log2);
        let refused = !gates_pass(units, t.spread_floor_log2) || ok < of;
        assert!(
            refused,
            "{phase} is refused in the table but re-derives as shippable"
        );
    }
}

// ---- 4. LOGO split correctness and the recorded evidence --------------------------------------

#[test]
fn priors_logo_holds_out_each_family_entirely() {
    let t = table();
    let man = manifest();
    for phase in t.shipped_phases() {
        let units = &man[phase];
        let l = logo(units, t.spread_floor_log2);
        let families: BTreeSet<&str> = units.iter().map(|u| u.family.as_str()).collect();
        assert_eq!(
            l.splits.len(),
            families.len(),
            "{phase}: one fold per family"
        );
        for (g, train, test) in &l.splits {
            assert!(!test.is_empty(), "{phase}/{g}: empty held-out fold");
            assert!(
                test.iter().all(|u| &u.family == g),
                "{phase}/{g}: a foreign unit was held out"
            );
            assert!(
                train.iter().all(|u| &u.family != g),
                "{phase}/{g}: the held-out family leaked into training"
            );
            let train_models: BTreeSet<&str> = train.iter().map(|u| u.model.as_str()).collect();
            assert!(
                test.iter()
                    .all(|u| !train_models.contains(u.model.as_str())),
                "{phase}/{g}: a held-out MODEL is also in training"
            );
            assert_eq!(
                train.len() + test.len(),
                units.len(),
                "{phase}/{g}: folds must partition"
            );
            let w: f64 = train.iter().map(|u| u.weight).sum();
            close(
                w,
                (families.len() - 1) as f64,
                1e-9,
                &format!("{phase}/{g}: re-balanced weight"),
            );
        }
        let ev = t
            .evidence
            .iter()
            .find(|e| e.phase == phase)
            .unwrap_or_else(|| panic!("{phase}: no evidence"));
        assert_eq!(
            l.families as u64, ev.logo_families,
            "{phase}: held-out families"
        );
        assert_eq!(l.wins as u64, ev.wins, "{phase}: LOGO wins vs the null");
        assert_eq!(
            l.losses as u64, ev.losses,
            "{phase}: LOGO losses vs the null"
        );
        close(
            l.gap_median,
            ev.gap_median,
            5e-6,
            &format!("{phase}: overfit gap"),
        );
        assert!(
            l.gap_median <= OVERFIT_GAP_MAX_LOG2,
            "{phase}: overfit gap over the bound"
        );
        assert!(
            l.wins >= l.losses,
            "{phase}: LOGO must not lose to the constant null"
        );
        let (ok, of) = stability(units, t.spread_floor_log2);
        assert_eq!(
            (ok as u64, of as u64),
            (ev.stable_folds, ev.stable_of),
            "{phase}: G7 stability re-derives"
        );
        assert_eq!(
            ok, of,
            "{phase}: the ship decision must survive removing any one family"
        );
    }
}

// ---- 5. the anti-fingerprint bound, with a positive control -----------------------------------

#[test]
fn priors_key_cannot_identify_family_beyond_bound_on_held_out_models() {
    let t = table();
    let man = manifest();
    for phase in t.shipped_phases() {
        let units = &man[phase];
        let shipped: BTreeSet<String> = t
            .models
            .iter()
            .filter(|m| m.phase == phase && m.cell != ROOT_CELL)
            .map(|m| m.cell.clone())
            .collect();
        let key = |u: &Unit| {
            if shipped.contains(&u.cell) {
                u.cell.clone()
            } else {
                ROOT_CELL.to_owned()
            }
        };
        let (acc, base) = classify(units, &key);
        let ev = t
            .evidence
            .iter()
            .find(|e| e.phase == phase)
            .expect("evidence");
        close(
            acc,
            ev.fp_accuracy,
            5e-6,
            &format!("{phase}: key-classifier accuracy"),
        );
        close(
            base,
            ev.fp_baseline,
            5e-6,
            &format!("{phase}: majority baseline"),
        );
        assert!(
            acc - base <= FINGERPRINT_EXCESS_MAX,
            "{phase}: the prior's input key recovers the family {acc:.3} vs baseline {base:.3} \
             (> {FINGERPRINT_EXCESS_MAX} excess): it would be a lookup by identity"
        );
        // POSITIVE CONTROL: under the SAME held-out-model protocol, a nearest-
        // neighbour classifier on the raw size (what the key deliberately is
        // not) recovers the family well beyond the bound, so a pass above is a
        // property of the key, not of a weak protocol.
        let (size_acc, size_base) = nearest_size(units);
        assert!(
            size_acc - size_base > FINGERPRINT_EXCESS_MAX + 0.05,
            "{phase}: positive control failed — size should fingerprint ({size_acc:.3} vs {size_base:.3})"
        );
        assert!(
            size_acc > acc,
            "{phase}: size must identify more than the key"
        );
    }
}

// ---- 6. purity / isolation --------------------------------------------------------------------

#[test]
fn priors_cells_span_two_families_within_purity() {
    let t = table();
    let man = manifest();
    for m in &t.models {
        assert!(
            m.families >= MIN_FAMILIES_PER_CELL,
            "{}[{}]",
            m.phase,
            m.cell
        );
        if m.cell != ROOT_CELL {
            let c = &census(&man[&m.phase])[&m.cell];
            assert!(
                c.families >= 2,
                "{}[{}] isolates one family",
                m.phase,
                m.cell
            );
            assert!(
                c.share_max <= PURITY_MAX,
                "{}[{}] share {}",
                m.phase,
                m.cell,
                c.share_max
            );
        }
    }
    // A cell that FAILS the gate collapses to the root (never ships).
    for (phase, units) in &man {
        let shipped: BTreeSet<&str> = t
            .models
            .iter()
            .filter(|m| &m.phase == phase)
            .map(|m| m.cell.as_str())
            .collect();
        for (cell, c) in census(units) {
            let ok = c.families >= 2 && c.share_max <= PURITY_MAX;
            if !ok {
                assert!(
                    !shipped.contains(cell.as_str()),
                    "{phase}[{cell}] failed G2 but ships"
                );
            }
        }
    }
}

// ---- 7. name-blind source ratchet ---------------------------------------------------------------

/// Every identifier-shaped string literal of the two track rosters.
fn roster() -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    for rel in [
        "../ny-cli/src/commands/vnncomp_2025_tracks.rs",
        "../ny-cli/src/commands/vnncomp_2026_tracks.rs",
    ] {
        let text = read(rel);
        for chunk in text.split('"').skip(1).step_by(2) {
            if chunk.len() >= 4
                && chunk
                    .bytes()
                    .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_')
                && chunk.bytes().any(|b| b.is_ascii_lowercase())
            {
                out.insert(chunk.to_owned());
            }
        }
    }
    out
}

#[test]
fn priors_outputs_are_not_keyed_by_name() {
    let ids = roster();
    assert!(
        ids.len() >= 24,
        "the roster scan found only {} ids",
        ids.len()
    );
    let table_text = COST_TABLE_V1.to_ascii_lowercase();
    let reader = read("src/router/priors.rs").to_ascii_lowercase();
    let fixture = read("tests/data/priors_v1_rows.tsv").to_ascii_lowercase();
    for id in &ids {
        for (what, text) in [
            ("table", &table_text),
            ("reader", &reader),
            ("fixture", &fixture),
        ] {
            assert!(
                !text.contains(id.as_str()),
                "the prior's {what} names `{id}`: priors are keyed by structure, never by name"
            );
        }
    }
    // POSITIVE CONTROL: the tooling's training set (which must name its
    // sources to hold families out) does contain roster ids, so the same scan
    // would see one that leaked into the prior.
    let trainset = std::fs::read_to_string(
        crate_root().join("../../scripts/router_priors/data/priors_v1_trainset.tsv"),
    )
    .expect("the committed training set")
    .to_ascii_lowercase();
    assert!(
        ids.iter()
            .filter(|id| trainset.contains(id.as_str()))
            .count()
            >= 10,
        "positive control: the training set names its sources"
    );
    // No path, no file name in any record (comment lines may cite the generator).
    for (what, text) in [("table", COST_TABLE_V1), ("fixture", &fixture)] {
        for line in text.lines().filter(|l| !l.starts_with('#')) {
            for needle in [
                ".onnx",
                ".vnnlib",
                "/home",
                "./",
                "\\",
                "benchmarks",
                "scratchpad",
            ] {
                assert!(
                    !line.contains(needle),
                    "{what} record carries `{needle}`: {line}"
                );
            }
            if !line.starts_with("schema\t") {
                assert!(
                    !line.contains('/'),
                    "{what} record carries a path separator: {line}"
                );
            }
        }
    }
    // The outputs are keyed ONLY by the closed key vocabulary.
    let t = table();
    for m in &t.models {
        assert!(priors::CELLS.contains(&m.cell.as_str()), "{}", m.cell);
        assert_eq!(m.work, priors::WORK_PARAMS);
    }
    for line in fixture.lines().filter(|l| !l.starts_with('#')) {
        let f: Vec<&str> = line.split('\t').collect();
        assert!(
            f[1].len() == 3
                && f[1].starts_with('f')
                && f[1][1..].bytes().all(|b| b.is_ascii_digit()),
            "fixture family {:?} must be an OPAQUE id",
            f[1]
        );
        assert!(priors::CELLS.contains(&f[3]), "fixture cell {:?}", f[3]);
    }
}

// ---- 8. predictions: monotone, host-gated, fail closed ------------------------------------------

fn gb10() -> HostEnvelope {
    HostEnvelope::declared_target(HostClass::Gb10)
}

fn facts(params: u64, ops: OpSet) -> InstanceFacts {
    let mut f = InstanceFacts::known_empty();
    f.params = Some(params);
    f.ops = ops;
    f.ops_complete = true;
    f
}

#[test]
fn priors_predictions_are_monotone_host_gated_and_fail_closed() {
    let t = table();
    let mlp = OpSet::EMPTY.with(OpKind::Linear).with(OpKind::Relu);
    let mut last = 0;
    for exp in [8_u32, 12, 16, 20, 24, 27] {
        let p = t
            .predict("verifier.setup", &facts(1 << exp, mlp), &gb10())
            .expect("priced on the measured host");
        assert!(p.median_ms >= last, "the shared exponent is monotone");
        assert!(p.upper_ms >= p.median_ms);
        last = p.median_ms;
    }
    let m5 = HostEnvelope::declared_target(HostClass::M5);
    assert_eq!(
        t.predict("verifier.setup", &facts(1 << 16, mlp), &m5),
        Err("host:m5".to_owned()),
        "a GB10 prior says nothing about an m5"
    );
    assert_eq!(
        t.predict(
            "verifier.setup",
            &InstanceFacts::unknown("scan>256MiB"),
            &gb10()
        ),
        Err("facts-unknown".to_owned())
    );
    let mut no_params = facts(1, mlp);
    no_params.params = None;
    assert_eq!(
        t.predict("verifier.setup", &no_params, &gb10()),
        Err("params-unknown".to_owned())
    );
    assert_eq!(
        t.predict("root.bound", &facts(1 << 16, mlp), &gb10()),
        Err("root.bound:refused".to_owned())
    );
    assert_eq!(
        t.predict("attack.upfront", &facts(1 << 16, mlp), &gb10()),
        Err("attack.upfront:refused".to_owned())
    );
    // An incomplete signature reads the root model (the nonpwl bit would be a guess).
    let mut partial = facts(1 << 16, mlp);
    partial.ops_complete = false;
    assert_eq!(CostKey::from_facts(&partial), None);
    assert_eq!(
        t.predict("verifier.setup", &partial, &gb10())
            .expect("root")
            .cell,
        ROOT_CELL
    );
    let conv = OpSet::EMPTY.with(OpKind::Conv).with(OpKind::Relu);
    assert_eq!(
        CostKey::from_facts(&facts(1, conv))
            .map(CostKey::token)
            .as_deref(),
        Some("conv=1;nonpwl=0")
    );
    let attn = OpSet::EMPTY.with(OpKind::Bilinear).with(OpKind::Softmax);
    assert_eq!(
        CostKey::from_facts(&facts(1, attn))
            .map(CostKey::token)
            .as_deref(),
        Some("conv=0;nonpwl=1")
    );
}

#[test]
fn priors_view_prices_only_bindings_whose_every_phase_ships() {
    let t = table();
    let mlp = OpSet::EMPTY.with(OpKind::Linear).with(OpKind::Relu);
    let view = t.view(&facts(1 << 16, mlp), &gb10());
    assert_eq!(view.version, t.version_token());
    assert_eq!(view.cell.as_deref(), Some("conv=0;nonpwl=0"));
    assert!(
        view.predictions
            .iter()
            .any(|p| p.phase == "verifier.setup" && p.cell == "conv=0;nonpwl=0"),
        "the shipped phase is predicted: {view:?}"
    );
    assert!(view.priced.is_empty(), "v1 prices no first rung: {view:?}");
    assert!(view
        .unpriced
        .iter()
        .any(|u| u.cap == "attack.upfront_apgd" && u.reason == "attack.upfront:refused"));
    for cap in ["relu_split.graph", "input_split", "lowdim", "mip.ay"] {
        assert!(
            view.unpriced
                .iter()
                .any(|u| u.cap == cap && u.reason == "root.bound:refused"),
            "{cap} needs root.bound, which the gates refused: {view:?}"
        );
    }
    let elsewhere = t.view(
        &facts(1 << 16, mlp),
        &HostEnvelope::declared_target(HostClass::M5),
    );
    assert!(
        elsewhere.priced.is_empty(),
        "nothing is priced off the measured host"
    );
    assert_eq!(elsewhere.predictions.len(), 0);
    assert!(elsewhere.unpriced.iter().all(|u| u.reason == "host:m5"));
}

/// A synthetic table whose every bound phase ships: the reader PRICES its
/// capability (the arithmetic the compiled v1 table never reaches, since no
/// v1 binding has every phase shipped).
#[test]
fn priors_reader_prices_a_binding_whose_every_phase_ships() {
    let text = synthetic_table();
    let t = PriorTable::parse(&text).expect("the synthetic table parses");
    assert!(t.validate().is_empty(), "{:?}", t.validate());
    let mlp = OpSet::EMPTY.with(OpKind::Linear).with(OpKind::Relu);
    let row = facts(1 << 16, mlp);
    // log2 secs = -10 + 0.5 * 16 = -2 -> 250 ms; upper + 1.0 * 1.0 -> 500 ms.
    let p = t.predict("x.setup", &row, &gb10()).expect("priced");
    assert_eq!(
        (p.median_ms, p.upper_ms, p.cell.as_str()),
        (250, 500, "conv=0;nonpwl=0")
    );
    // The root answers a key with no cell of its own: -9 + 8 = -1 -> 500 ms.
    let conv = facts(1 << 16, OpSet::EMPTY.with(OpKind::Conv));
    assert_eq!(
        t.predict("x.setup", &conv, &gb10())
            .expect("root")
            .median_ms,
        500
    );
    assert_eq!(t.first_rung_ms("cap.x", &row, &gb10()), Ok(500));
    assert_eq!(
        t.first_rung_ms("cap.none", &row, &gb10()),
        Err("unbound".to_owned())
    );
    assert_eq!(t.view(&row, &gb10()).rung_ms("cap.x"), Some(500));
}

fn synthetic_table() -> String {
    let sha = format!("trainset_sha256\t{}", "0".repeat(64));
    let lines = [
        "schema\tny-router-priors/v1",
        "version\tcost-v1",
        "host\tgb10",
        sha.as_str(),
        "receipts\t10",
        "spread_k\t1.000000",
        "spread_floor_log2\t0.500000",
        "layer_b\tuniform\tsynthetic",
        "gate\tmin_families_per_cell\t2",
        "gate\tpurity_max\t0.600000",
        "gate\tfingerprint_excess_max\t0.200000",
        "gate\tslope_min\t0.000000",
        "gate\tslope_max\t2.000000",
        "gate\toverfit_gap_max_log2\t0.500000",
        "evidence\tx.setup\tlogo_families=3\tgap_median=0.100000\theldout_median=0.500000\t\
         null_median=1.000000\twins=3\tlosses=0\tsign_p=0.250000\tfp_accuracy=0.400000\t\
         fp_baseline=0.333333\tfp_excess=0.066667\tisolating=0\tstable_folds=3\tstable_of=3",
        "model\tx.setup\tsecs\tparams\t*\t-9.000000\t0.500000\t1.000000\t9\t3\t0.333333",
        "model\tx.setup\tsecs\tparams\tconv=0;nonpwl=0\t-10.000000\t0.500000\t1.000000\t6\t2\t\
         0.500000",
        "bind\tcap.x\tx.setup",
    ];
    lines.join("\n") + "\n"
}

// ---- 9. the plan: dark without a prior, conservative with one -----------------------------------

fn cap(id: &str, class: CapClass, admission: AdmissionState, rung: Option<u64>) -> CapInput {
    CapInput {
        id: id.to_owned(),
        class,
        ledger: LedgerState::SealedAsShipped,
        progress: ProgressKind::Margin,
        unit: WorkUnit::Restart,
        first_rung_ms: rung,
        safe_points: SafePoints::BetweenUnits,
        accepts: AmendmentSet::NONE,
        admission,
        route: None,
    }
}

/// A prior view that priced the attack lane at `rung` (the plan reads the
/// view and the capabilities' first rungs, never the table).
fn priced_view(rung: Option<u64>) -> PriorView {
    PriorView {
        version: "cost-v1+5c0ffee5".to_owned(),
        host: "gb10".to_owned(),
        cell: Some("conv=0;nonpwl=0".to_owned()),
        predictions: Vec::new(),
        priced: rung
            .map(|ms| PricedCap {
                cap: "attack.upfront_apgd".to_owned(),
                ms,
            })
            .into_iter()
            .collect(),
        unpriced: Vec::new(),
    }
}

fn input(official_ms: u64, attack_rung: Option<u64>, prior: bool) -> PolicyInput {
    let mlp = OpSet::EMPTY.with(OpKind::Linear).with(OpKind::Relu);
    let row = facts(1 << 16, mlp);
    let host = gb10();
    PolicyInput {
        schema: ROUTER_SCHEMA.to_owned(),
        facts: row,
        host,
        budget: BudgetView {
            official_ms,
            internal_ms: official_ms.saturating_sub(5_000),
            join_margin_ms: 2_000,
        },
        capabilities: vec![
            cap(
                "attack.upfront_apgd",
                CapClass::Lane(LaneRole::Attack),
                AdmissionState::Unknown("inline-admission".to_owned()),
                attack_rung,
            ),
            cap(
                "relu_split.graph",
                CapClass::Lane(LaneRole::Prover { terminal_ok: true }),
                AdmissionState::Unknown("inline-admission".to_owned()),
                None,
            ),
            cap(
                "ibp-fast",
                CapClass::Lane(LaneRole::Screen),
                AdmissionState::Admit,
                None,
            ),
        ],
        pins: Vec::new(),
        priors_version: if prior {
            "cost-v1+5c0ffee5".to_owned()
        } else {
            PRIORS_VERSION.to_owned()
        },
        registry_version: "r7test00".to_owned(),
        priors: prior.then(|| priced_view(attack_rung)),
    }
}

#[test]
fn priors_absent_plan_and_hash_are_r5_bit_identical() {
    for official in [20_000_u64, 100_000, 900_000] {
        let none = input(official, None, false);
        let json = serde_json::to_value(&none).expect("serializes");
        assert!(
            json.get("priors").is_none(),
            "an input without a prior must serialize exactly R5's fields"
        );
        let back: PolicyInput = serde_json::from_value(json).expect("an R5 record deserializes");
        assert_eq!(back, none);
        assert_eq!(hash::static_hash(&back), hash::static_hash(&none));
        let p = plan::plan(&none);
        assert!(!p.rules_fired.iter().any(|r| r == RULE_PRIORS.as_str()));
        assert!(!p.would.iter().any(|w| w.rule == RULE_PRIORS.as_str()));
        assert!(
            p.contracts.iter().all(|c| c.cap == "ibp-fast"),
            "without a prior a deferred lane is never contracted: {:?}",
            p.contracts
        );
        let with = input(official, None, true);
        assert_ne!(
            hash::static_hash(&with),
            hash::static_hash(&none),
            "a read prior is a static input and must move the hash"
        );
    }
}

/// The static hash of an input WITHOUT a prior is R5's byte encoding: the
/// golden value below was computed by an independent re-encoding of R5's
/// documented layout (`ny-router/static-hash/v1`, length-prefixed strings,
/// little-endian u64s, one tag byte per option / enum / bool) — so a change
/// that encodes anything for an absent prior fails here.
///
/// #admission-precision (merged in int/eng13) appended ONE facts field,
/// `spec_shape` (tag 0 when absent), to that layout. The same independent
/// re-encoding reproduces R7's R5-layout golden
/// `924c3d049c1ea8bbeefeb7ca331cf23c7999b59283b6c6545a0550eefd1432f0`
/// without that byte, and gives the value below with it; the prior still
/// encodes nothing when absent.
#[test]
fn priors_absent_static_hash_is_the_r5_encoding() {
    let input = PolicyInput {
        schema: ROUTER_SCHEMA.to_owned(),
        facts: InstanceFacts::known_empty(),
        host: gb10(),
        budget: BudgetView {
            official_ms: 100_000,
            internal_ms: 95_000,
            join_margin_ms: 2_000,
        },
        capabilities: Vec::new(),
        pins: Vec::new(),
        priors_version: PRIORS_VERSION.to_owned(),
        registry_version: "r7test00".to_owned(),
        priors: None,
    };
    assert_eq!(
        hash::hex(&hash::static_hash(&input)),
        "d88915e312bd551568cfa5582fb30d9e7098100f091428cd6afd162c9a1dd0bb"
    );
}

/// Without a prior, a deferred lane is never contracted EVEN IF a rung were
/// set on it (only the prior's presence admits deferred lanes to CR-C).
#[test]
fn priors_absent_never_contracts_a_deferred_lane() {
    let mut none = input(100_000, Some(1_234), false);
    none.priors = None;
    let p = plan::plan(&none);
    assert!(
        p.contracts.iter().all(|c| c.cap != "attack.upfront_apgd"),
        "{:?}",
        p.contracts
    );
    assert!(p
        .declined
        .iter()
        .all(|(c, why)| c != "attack.upfront_apgd" || !why.starts_with("rung_does_not_fit")));
}

#[test]
fn priors_priced_deferred_lane_is_contracted_and_an_unfit_rung_keeps_todays_deadline() {
    let rung = 4_321;
    // Fits: contracted at its rung, the attack slot stays governed.
    let fits = input(100_000, Some(rung), true);
    let p = plan::plan(&fits);
    assert!(p.rules_fired.iter().any(|r| r == RULE_PRIORS.as_str()));
    let c = p
        .contracts
        .iter()
        .find(|c| c.cap == "attack.upfront_apgd")
        .expect("a priced deferred lane is contracted");
    assert_eq!(c.ms, Some(rung));
    assert!(!c.terminal);
    assert_eq!(
        p.terminal, None,
        "a deferred prover never becomes the terminal"
    );
    assert!(
        p.would
            .iter()
            .any(|w| w.rule == RULE_PRIORS.as_str()
                && w.what.contains("priced=[attack.upfront_apgd:"))
    );
    let fitting = acting::commit(&p, &fits, EffectiveMode::Acting, &RuleSwitches::ALL_ON);
    assert!(
        fitting.governed(LaneSlot::AttackUpfront),
        "{:?}",
        fitting.governance
    );
    // Does not fit: declined, and CR-A leaves the slot on today's deadline.
    let unfit = input(100_000, Some(10_000_000), true);
    let p = plan::plan(&unfit);
    assert!(
        p.declined
            .iter()
            .any(|(c, why)| c == "attack.upfront_apgd" && why.starts_with("rung_does_not_fit")),
        "{:?}",
        p.declined
    );
    let refused = acting::commit(&p, &unfit, EffectiveMode::Acting, &RuleSwitches::ALL_ON);
    let g = refused
        .governance
        .iter()
        .find(|g| g.slot == LaneSlot::AttackUpfront)
        .expect("the attack slot");
    assert!(
        !g.governed,
        "an unfit priced rung must leave today's deadline"
    );
    assert_eq!(g.reason, "rung");
    // Shadow never acts on it.
    let shadow = acting::commit(&p, &unfit, EffectiveMode::Shadow, &RuleSwitches::ALL_ON);
    assert!(!shadow.acting && shadow.governance.is_empty());
}

#[test]
fn priors_plan_is_pure_with_a_prior() {
    let a = input(100_000, Some(1_234), true);
    let b = a.clone();
    assert_eq!(plan::plan(&a), plan::plan(&b));
    assert_eq!(hash::static_hash(&a), hash::static_hash(&b));
    // DeviceQual is irrelevant to the prior; the host CLASS gates it.
    let mut other = a.clone();
    other.host.device = DeviceQual::None;
    assert_eq!(plan::plan(&other).rules_fired, plan::plan(&a).rules_fired);
}

// ---- 10. review (#router-r7): what the bound covers, and three checks with no test before ----------

/// Review (#router-r7). The G3 bound is a property of the prior's KEY. Its
/// full INPUT (key + size) and therefore the OUTPUT every row's view logs
/// (`[router] priors predict=`, hashed into the policy input) DO identify the
/// family on held-out models: the prediction is a monotone map of the size
/// inside each cell, and the size fingerprints (the positive control above).
/// Measured on the v1 manifest: 1-NN on the logged median prediction of
/// `verifier.setup` beats the majority baseline by ~0.35, above the 0.20 bound.
/// So what keeps a size-bearing prior from being a lookup by identity is its
/// CAPACITY — one exponent per phase and at most one intercept per cell of
/// the closed key vocabulary — validated out of family by G6/G7, not
/// blindness. This test pins both halves so no document can drift into
/// claiming the outputs are identity-blind, and so a table that adds capacity
/// fails here.
#[test]
fn priors_outputs_carry_the_size_fingerprint_so_the_guard_is_capacity() {
    let t = table();
    let man = manifest();
    let mut checked = 0;
    for phase in t.shipped_phases() {
        let models: Vec<&priors::CostModel> =
            t.models.iter().filter(|m| m.phase == phase).collect();
        let slope = models[0].slope;
        assert!(
            models.iter().all(|m| m.slope.to_bits() == slope.to_bits()),
            "{phase}: one exponent per phase"
        );
        assert!(
            models.len() <= priors::CELLS.len(),
            "{phase}: at most one intercept per closed-vocabulary cell"
        );
        if slope < 0.05 {
            // A size-free phase carries no size fingerprint to pin.
            continue;
        }
        let logged: Vec<Unit> = man[phase]
            .iter()
            .map(|u| {
                let m = models
                    .iter()
                    .find(|m| m.cell == u.cell)
                    .or_else(|| models.iter().find(|m| m.cell == ROOT_CELL))
                    .expect("a shipped phase has a root model");
                Unit {
                    x: m.intercept + m.slope * u.x,
                    ..u.clone()
                }
            })
            .collect();
        let (acc, base) = nearest_size(&logged);
        assert!(
            acc - base > FINGERPRINT_EXCESS_MAX,
            "{phase}: the logged prediction no longer identifies the family ({acc:.3} vs \
             {base:.3}); if that is real, re-derive the anti-fingerprint claims before relaxing this"
        );
        checked += 1;
    }
    assert!(checked >= 1, "no shipped size-bearing phase was checked");
}

/// A synthetic table with a per-clause phase and a non-time unit beside the
/// `x.setup` phase of [`synthetic_table`].
fn synthetic_table_with_units() -> String {
    let extra = [
        "evidence\ty.bound\tlogo_families=3\tgap_median=0.100000\theldout_median=0.500000\t\
         null_median=1.000000\twins=3\tlosses=0\tsign_p=0.250000\tfp_accuracy=0.400000\t\
         fp_baseline=0.333333\tfp_excess=0.066667\tisolating=0\tstable_folds=3\tstable_of=3",
        "model\ty.bound\tsecs_per_clause\tparams\t*\t-12.000000\t0.500000\t1.000000\t9\t3\t0.333333",
        "evidence\tz.mem\tlogo_families=3\tgap_median=0.100000\theldout_median=0.500000\t\
         null_median=1.000000\twins=3\tlosses=0\tsign_p=0.250000\tfp_accuracy=0.400000\t\
         fp_baseline=0.333333\tfp_excess=0.066667\tisolating=0\tstable_folds=3\tstable_of=3",
        "model\tz.mem\tbytes\tparams\t*\t10.000000\t0.500000\t1.000000\t9\t3\t0.333333",
        "bind\tcap.y\tx.setup+y.bound",
        "bind\tcap.z\tz.mem",
    ];
    synthetic_table() + &extra.join("\n") + "\n"
}

/// Review (#router-r7): no test priced a `secs_per_clause` phase (v1 refuses
/// `root.bound`, the only one), so the clause scaling and its fail-closed arms
/// were unpinned. A per-clause phase is scaled by the row's clauses, refuses
/// an unknown or zero count, and a non-time unit never prices a rung.
#[test]
fn priors_reader_scales_per_clause_phases_and_refuses_non_time_units() {
    let t = PriorTable::parse(&synthetic_table_with_units()).expect("parses");
    assert!(t.validate().is_empty(), "{:?}", t.validate());
    let mlp = OpSet::EMPTY.with(OpKind::Linear).with(OpKind::Relu);
    let mut row = facts(1 << 16, mlp);
    row.clauses = Some(4);
    // log2 secs/clause = -12 + 0.5 * 16 = -4 -> 62.5 ms x 4 = 250 ms;
    // upper + 1.0 -> 125 ms x 4 = 500 ms.
    let p = t.predict("y.bound", &row, &gb10()).expect("priced");
    assert_eq!((p.median_ms, p.upper_ms), (250, 500));
    row.clauses = Some(1);
    assert_eq!(
        t.predict("y.bound", &row, &gb10())
            .expect("one clause")
            .median_ms,
        63,
        "62.5 ms ceils to 63"
    );
    for clauses in [None, Some(0)] {
        row.clauses = clauses;
        assert_eq!(
            t.predict("y.bound", &row, &gb10()),
            Err("y.bound:clauses-unknown".to_owned()),
            "{clauses:?}"
        );
        assert_eq!(
            t.first_rung_ms("cap.y", &row, &gb10()),
            Err("y.bound:clauses-unknown".to_owned()),
            "a binding with an unpriceable phase is unpriced"
        );
    }
    row.clauses = Some(4);
    // x.setup upper 500 ms + y.bound upper 500 ms.
    assert_eq!(t.first_rung_ms("cap.y", &row, &gb10()), Ok(1_000));
    assert_eq!(
        t.predict("z.mem", &row, &gb10()),
        Err("z.mem:unit-bytes-is-not-time".to_owned())
    );
    let view = t.view(&row, &gb10());
    assert_eq!(view.rung_ms("cap.y"), Some(1_000));
    assert_eq!(
        view.rung_ms("cap.z"),
        None,
        "a bytes model never prices a rung"
    );
    assert!(view
        .unpriced
        .iter()
        .any(|u| u.cap == "cap.z" && u.reason == "z.mem:unit-bytes-is-not-time"));
}

/// Review (#router-r7): the prior's view is a STATIC input (hashed so replay
/// and receipts can prove which prior a plan read), but no test moved any of
/// its fields one at a time — a dropped field would hash two different views
/// identically. Every field, and the order of each list, moves the hash.
#[test]
fn priors_view_every_field_moves_the_static_hash() {
    let mut base = input(100_000, Some(1_234), true);
    {
        let view = base.priors.as_mut().expect("the view");
        view.predictions = vec![
            PhasePrediction {
                phase: "verifier.setup".to_owned(),
                cell: ROOT_CELL.to_owned(),
                median_ms: 10,
                upper_ms: 20,
            },
            PhasePrediction {
                phase: "x.setup".to_owned(),
                cell: "conv=0;nonpwl=0".to_owned(),
                median_ms: 30,
                upper_ms: 40,
            },
        ];
        view.unpriced = vec![
            UnpricedCap {
                cap: "relu_split.graph".to_owned(),
                reason: "root.bound:refused".to_owned(),
            },
            UnpricedCap {
                cap: "lowdim".to_owned(),
                reason: "root.bound:refused".to_owned(),
            },
        ];
    }
    let h0 = hash::static_hash(&base);
    type Edit = fn(&mut PriorView);
    let edits: [(&str, Edit); 15] = [
        ("version", |v| v.version.push('0')),
        ("host", |v| v.host = "m5".to_owned()),
        ("cell", |v| v.cell = None),
        ("prediction phase", |v| v.predictions[0].phase.push('x')),
        ("prediction cell", |v| {
            v.predictions[0].cell = "conv=1;nonpwl=0".to_owned();
        }),
        ("prediction median", |v| v.predictions[0].median_ms += 1),
        ("prediction upper", |v| v.predictions[0].upper_ms += 1),
        ("prediction order", |v| v.predictions.swap(0, 1)),
        ("prediction dropped", |v| {
            v.predictions.pop();
        }),
        ("priced cap", |v| v.priced[0].cap.push('x')),
        ("priced ms", |v| v.priced[0].ms += 1),
        ("unpriced cap", |v| v.unpriced[0].cap.push('x')),
        ("unpriced reason", |v| {
            v.unpriced[0].reason = "host:m5".to_owned();
        }),
        ("unpriced order", |v| v.unpriced.swap(0, 1)),
        ("unpriced dropped", |v| v.unpriced.clear()),
    ];
    for (what, edit) in edits {
        let mut moved = base.clone();
        edit(moved.priors.as_mut().expect("the view"));
        assert_ne!(
            moved.priors, base.priors,
            "{what}: the edit must change the view"
        );
        assert_ne!(
            hash::static_hash(&moved),
            h0,
            "{what}: two different prior views hash identically"
        );
    }
}

/// Review (#router-r7): under a prior, CR-C contracts the admitted lanes and
/// the PRICED deferred lanes — never an unpriced deferred one (its own
/// predicate has not answered and the prior said nothing about it). And a
/// prior that prices nothing contracts exactly what R5 contracts.
#[test]
fn priors_contract_only_priced_deferred_lanes() {
    let p = plan::plan(&input(100_000, Some(4_321), true));
    let contracted: BTreeSet<&str> = p.contracts.iter().map(|c| c.cap.as_str()).collect();
    let want: BTreeSet<&str> = ["attack.upfront_apgd", "ibp-fast"].into_iter().collect();
    assert_eq!(contracted, want, "{:?}", p.contracts);
    assert!(
        !p.order.iter().any(|c| c == "relu_split.graph"),
        "an unpriced deferred lane is never contracted: {:?}",
        p.order
    );
    let priced_nothing = plan::plan(&input(100_000, None, true));
    let r5 = plan::plan(&input(100_000, None, false));
    assert_eq!(priced_nothing.contracts, r5.contracts);
    assert_eq!(priced_nothing.order, r5.order);
    assert_eq!(priced_nothing.declined, r5.declined);
    assert_eq!(priced_nothing.terminal, r5.terminal);
}
