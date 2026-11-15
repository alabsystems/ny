// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #admission-census unit tests (run with `--bin ny`, never `--lib`).
//!
//! Fixtures are hermetic: a synthetic margin-row-family net (the
//! `proof_check` builder's shape: 2x6x6 input, a residual block, two trunk
//! ReLUs, a 5-wide head, 4 classes) and properties written into a temporary
//! directory, plus the tracked `tests/models/simple_mlp.onnx`. No
//! `benchmarks/` dependency.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use ny_core::router::decl::{OpKind, OpSet, SpecSet};
use ny_core::router::facts::HostClass;
use ny_core::router::route::CENSUS_WIDEST_ENVELOPE;
use ny_onnx::onnx_proto::tensor_shape_proto::{dimension, Dimension};
use ny_onnx::onnx_proto::{
    AttributeProto, GraphProto, ModelProto, NodeProto, TensorProto, TensorShapeProto,
    TensorTypeProto, TypeProto, ValueInfoProto,
};
use prost::Message;

use super::census::{
    self, census_text, parse_corpus, parse_envelope, verify_census_text, CensusOptions, CorpusRow,
    COLUMNS, CORPUS_SCHEMA,
};
use super::{policy_input, Authority, StartContext};
use crate::commands::margin_row_bab::{lane_model_admission, lane_property_admission};
use crate::plan_resolver::ModelFacts;

pub(super) fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

pub(super) fn read(rel: &str) -> String {
    let path = workspace_root().join(rel);
    let text = fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    assert!(
        text.len() > 200,
        "{} is suspiciously short: the check would be VOID",
        path.display()
    );
    text
}

pub(super) fn code_only(text: &str) -> String {
    text.lines()
        .filter(|line| !line.trim_start().starts_with("//"))
        .collect::<Vec<_>>()
        .join("\n")
}

/// The body of `fn <name>(` in `text`: from its signature to the first line
/// that closes an item at column 0.
pub(super) fn fn_body<'a>(text: &'a str, name: &str) -> &'a str {
    let start = text
        .find(&format!("fn {name}("))
        .unwrap_or_else(|| panic!("fn {name} not found: the check would be VOID"));
    let rest = &text[start..];
    let end = rest.find("\n}\n").map_or(rest.len(), |i| i + 3);
    &rest[..end]
}

// ---- fixtures -----------------------------------------------------------------

fn node(op: &str, inputs: &[&str], outputs: &[&str], attrs: Vec<AttributeProto>) -> NodeProto {
    NodeProto {
        op_type: op.to_owned(),
        input: inputs.iter().map(|s| (*s).to_owned()).collect(),
        output: outputs.iter().map(|s| (*s).to_owned()).collect(),
        attribute: attrs,
        ..Default::default()
    }
}

fn ints(name: &str, v: &[i64]) -> AttributeProto {
    AttributeProto {
        name: name.to_owned(),
        ints: v.to_vec(),
        r#type: 7,
        ..Default::default()
    }
}

fn int(name: &str, v: i64) -> AttributeProto {
    AttributeProto {
        name: name.to_owned(),
        i: Some(v),
        r#type: 2,
        ..Default::default()
    }
}

fn f32_tensor(name: &str, dims: &[i64], n: usize) -> TensorProto {
    TensorProto {
        name: name.to_owned(),
        dims: dims.to_vec(),
        data_type: 1,
        float_data: (0..n)
            .map(|i| ((i * 37 % 101) as f32 / 101.0 - 0.5) * 0.7)
            .collect(),
        ..Default::default()
    }
}

/// A margin-row-family net: 2x6x6 input, conv + residual block, two trunk
/// ReLUs, `Gemm -> ReLU -> Gemm` head with 4 classes. `heads` = 1 drops the
/// second Gemm (outside the compiled head).
pub(super) fn twin_model_bytes(heads: usize) -> Vec<u8> {
    let dim = |d: i64| Dimension {
        value: Some(dimension::Value::DimValue(d)),
    };
    let vi = |name: &str, dims: &[i64]| ValueInfoProto {
        name: name.to_owned(),
        r#type: Some(TypeProto {
            tensor_type: Some(TensorTypeProto {
                elem_type: 1,
                shape: Some(TensorShapeProto {
                    dim: dims.iter().map(|&d| dim(d)).collect(),
                }),
            }),
        }),
    };
    let mut initializer = vec![
        f32_tensor("c1w", &[4, 2, 3, 3], 72),
        f32_tensor("c1b", &[4], 4),
        f32_tensor("c2w", &[4, 4, 3, 3], 144),
        f32_tensor("c2b", &[4], 4),
    ];
    let mut nodes = vec![
        node(
            "Conv",
            &["x", "c1w", "c1b"],
            &["c1"],
            vec![ints("strides", &[2, 2]), ints("pads", &[1, 1, 1, 1])],
        ),
        node("Relu", &["c1"], &["r1"], Vec::new()),
        node(
            "Conv",
            &["r1", "c2w", "c2b"],
            &["c2"],
            vec![ints("pads", &[1, 1, 1, 1])],
        ),
        node("Add", &["c2", "r1"], &["a1"], Vec::new()),
        node("Relu", &["a1"], &["r2"], Vec::new()),
        node("Flatten", &["r2"], &["f"], vec![int("axis", 1)]),
    ];
    let out_width = if heads == 2 {
        initializer.extend([
            f32_tensor("g1w", &[5, 36], 180),
            f32_tensor("g1b", &[5], 5),
            f32_tensor("g2w", &[4, 5], 20),
            f32_tensor("g2b", &[4], 4),
        ]);
        nodes.extend([
            node(
                "Gemm",
                &["f", "g1w", "g1b"],
                &["g1"],
                vec![int("transB", 1)],
            ),
            node("Relu", &["g1"], &["h"], Vec::new()),
            node("Gemm", &["h", "g2w", "g2b"], &["y"], vec![int("transB", 1)]),
        ]);
        4
    } else {
        initializer.extend([f32_tensor("g1w", &[4, 36], 144), f32_tensor("g1b", &[4], 4)]);
        nodes.push(node(
            "Gemm",
            &["f", "g1w", "g1b"],
            &["y"],
            vec![int("transB", 1)],
        ));
        4
    };
    ModelProto {
        ir_version: 8,
        graph: Some(GraphProto {
            node: nodes,
            initializer,
            input: vec![vi("x", &[1, 2, 6, 6])],
            output: vec![vi("y", &[1, out_width])],
            ..Default::default()
        }),
        ..Default::default()
    }
    .encode_to_vec()
}

/// An argmax-robustness property over `n_x` inputs and 4 classes (true 0).
pub(super) fn robustness_property(n_x: usize) -> String {
    let mut s = String::new();
    for i in 0..n_x {
        s.push_str(&format!("(declare-const X_{i} Real)\n"));
    }
    for j in 0..4 {
        s.push_str(&format!("(declare-const Y_{j} Real)\n"));
    }
    for i in 0..n_x {
        s.push_str(&format!(
            "(assert (>= X_{i} {:?}))\n(assert (<= X_{i} {:?}))\n",
            0.1_f64, 0.2_f64
        ));
    }
    s.push_str("(assert (or\n");
    for j in 1..4 {
        s.push_str(&format!("    (and (>= Y_{j} Y_0))\n"));
    }
    s.push_str("))\n");
    s
}

/// A single half-space property (not robustness) over `n_x` inputs.
pub(super) fn halfspace_property(n_x: usize) -> String {
    let mut s = String::new();
    for i in 0..n_x {
        s.push_str(&format!("(declare-const X_{i} Real)\n"));
    }
    for j in 0..4 {
        s.push_str(&format!("(declare-const Y_{j} Real)\n"));
    }
    for i in 0..n_x {
        s.push_str(&format!(
            "(assert (>= X_{i} 0.1))\n(assert (<= X_{i} 0.2))\n"
        ));
    }
    s.push_str("(assert (<= Y_0 -1000000.0))\n");
    s
}

/// The 2026 isomorphic shape: TWO declared networks (the relational route).
const RELATIONAL: &str = "(vnnlib-version <2.0>)
(declare-network f
    (declare-input X_f real [1, 1, 1, 5])
    (declare-output Y_f real [1, 5])
)
(declare-network g
    (isomorphic-to f)
    (declare-input X_g real [1, 1, 1, 5])
    (declare-output Y_g real [1, 5])
)
(assert (and (<= X_f[0,0,0,0] 0.6) (>= X_f[0,0,0,0] 0.5)))
";

pub(super) struct World {
    pub(super) dir: tempfile::TempDir,
}

impl World {
    pub(super) fn new() -> Self {
        Self {
            dir: tempfile::tempdir().expect("temporary census world"),
        }
    }

    pub(super) fn file(&self, rel: &str, bytes: &[u8]) -> PathBuf {
        let path = self.dir.path().join(rel);
        fs::create_dir_all(path.parent().expect("parent")).expect("mkdir");
        fs::write(&path, bytes).expect("write fixture");
        path
    }

    fn gz(&self, rel: &str, bytes: &[u8]) -> PathBuf {
        use std::io::Write as _;
        let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        encoder.write_all(bytes).expect("gzip");
        self.file(rel, &encoder.finish().expect("gzip finish"))
    }

    pub(super) fn options(&self, tag: &str) -> CensusOptions {
        CensusOptions {
            manifest: self.dir.path().join(format!("{tag}.corpus.tsv")),
            out: self.dir.path().join(format!("{tag}.census.tsv")),
            envelope: CENSUS_WIDEST_ENVELOPE,
            // A declared host: no host or device probe in a unit test.
            target_host: Some(HostClass::M5),
            row_deadline: Duration::from_secs(60),
            work_dir: self.dir.path().join(format!("{tag}.work")),
        }
    }
}

pub(super) fn row(id: &str, model: &Path, property: &Path) -> CorpusRow {
    CorpusRow {
        row_id: id.to_owned(),
        model: model.to_path_buf(),
        property: property.to_path_buf(),
        budget_secs: 100,
    }
}

/// The census of `rows`: `(row_id, cap) -> [admit, router, parent_probe,
/// reasons, facts, ops, form]`.
pub(super) fn census_cells(
    rows: &[CorpusRow],
    options: &CensusOptions,
) -> (String, Vec<(String, String, Vec<String>)>) {
    fs::create_dir_all(&options.work_dir).expect("work dir");
    let (text, _) = census_text(rows, "c0ffee", "b1", "commit=test", options);
    let cells = text
        .lines()
        .filter(|line| !line.starts_with('#'))
        .map(|line| {
            let f: Vec<&str> = line.split('\t').collect();
            assert_eq!(f.len(), COLUMNS.len(), "{line}");
            (
                f[0].to_owned(),
                f[1].to_owned(),
                f[2..].iter().map(|s| (*s).to_owned()).collect(),
            )
        })
        .collect();
    (text, cells)
}

pub(super) fn cell<'a>(
    cells: &'a [(String, String, Vec<String>)],
    row_id: &str,
    cap: &str,
) -> &'a [String] {
    &cells
        .iter()
        .find(|(r, c, _)| r == row_id && c == cap)
        .unwrap_or_else(|| panic!("no census cell for {row_id}/{cap}"))
        .2
}

// ---- tests --------------------------------------------------------------------

/// SHARED FUNCTION: a census row's router input IS the live router's input
/// — `policy_input` over the plan resolver's own scan
/// (`ModelFacts::from_onnx_file`) and the on-disk size, exactly as the scored
/// entry point's start context carries them — and a gzip-compressed model
/// yields the facts of its competition (plain) form.
#[test]
fn census_facts_are_the_live_router_facts() {
    let world = World::new();
    let model = world.file("m/net.onnx", &twin_model_bytes(2));
    let property = world.file("p/prop.vnnlib", robustness_property(72).as_bytes());
    let options = world.options("facts");
    let (_, cells) = census_cells(&[row("r1", &model, &property)], &options);
    let scan = ModelFacts::from_onnx_file(&model).expect("the scan reads the fixture");
    // #prepared-root-route: the census's own lever context (a row with no
    // category file, zero NY_*), which `prepared-root`'s predicate reads.
    let levers = census::no_category_levers();
    let live = policy_input(&StartContext {
        levers: Some(&levers),
        authority: Authority::Frozen,
        model: Some(&scan),
        model_bytes: fs::metadata(&model).ok().map(|m| m.len()),
        facts_reason: None,
        declared_networks: Some(0),
        relational_selected: Some(false),
        official_budget_secs: 100,
        internal_secs: 100,
        join_margin: crate::commands::vnncomp::CONCURRENT_LANE_JOIN_MARGIN,
        anchor: Instant::now(),
        target_host: Some(HostClass::M5),
        pinned_keys: Vec::new(),
        trial_pinned_keys: Vec::new(),
        // #admission-precision: the entry point's own property read.
        spec: crate::commands::beta_crown::spec_shape::read_spec(&property),
    });
    assert!(live.facts.is_known() && live.facts.ops_complete);
    let expect_ops = OpSet::EMPTY
        .with(OpKind::Conv)
        .with(OpKind::Relu)
        .with(OpKind::Residual)
        .with(OpKind::Linear);
    assert_eq!(live.facts.ops, expect_ops);
    let fd = cell(&cells, "r1", "frozen-dual-bab");
    assert_eq!(fd[4], "known/complete");
    assert_eq!(fd[5], "linear+conv+relu+residual");
    // Every census decision equals the router's own decision on `live`.
    for cap in ny_core::router::route::census_capabilities(&live) {
        let d = ny_core::router::route::census_decision(&live, cap, CENSUS_WIDEST_ENVELOPE)
            .expect("decided");
        assert_eq!(
            cell(&cells, "r1", cap)[1],
            if d.routed { "1" } else { "0" },
            "{cap}: router column = census_decision on the live input"
        );
    }
    // The competition form: the same net gzip-compressed has the same facts
    // and the same decisions (only the `form` column differs).
    let gz_model = world.gz("m/net.onnx.gz", &twin_model_bytes(2));
    let (_, gz_cells) = census_cells(&[row("r1", &gz_model, &property)], &world.options("gz"));
    for (r, c, f) in &cells {
        let g = cell(&gz_cells, r, c);
        assert_eq!(&f[..7], &g[..7], "{c}: gz vs plain");
        assert_eq!(f[7], "plain");
        assert_eq!(g[7], "gz");
    }
    // #admission-precision: the `spec` column is the entry point's read.
    let read = crate::commands::beta_crown::spec_shape::read_spec(&property);
    assert_eq!(fd[6], read.token());
    assert_eq!(live.facts.spec, read.archetype());
    assert_eq!(live.facts.spec_shape, read.shape());
    assert!(fd[6].starts_with("robustness "), "{}", fd[6]);
}

/// SOURCE RATCHET: the census asks the live extractors and the live
/// decision, never a copy of them, and the scored entry point still builds
/// its router start from exactly those calls.
#[test]
fn census_uses_the_live_facts_extractor() {
    let census = code_only(&read("crates/ny-cli/src/router/census.rs"));
    for needle in [
        "ModelFacts::from_onnx_file(path)",
        "RelationalRouteDecision::for_vnnlib(&row.property)",
        // #prepared-root-route: with the no-category competition-form levers.
        "let (host, levers) = (self.target_host, &self.levers);",
        "policy_input(&start_context(row, declared, host, levers, start))",
        "levers: no_category_levers(),",
        "route_policy::census_decision(&input, cap, envelope)",
        "super::facts::host_envelope(options.target_host)",
        // #admission-precision: the entry point's own property read.
        ".or_insert_with(|| read_spec(path))",
    ] {
        assert!(census.contains(needle), "census.rs must call `{needle}`");
    }
    for needle in [
        "instance_facts(",
        "InstanceFacts::known_empty",
        "fn commit(",
    ] {
        assert!(
            !census.contains(needle),
            "census.rs must not build facts or decide itself (`{needle}`)"
        );
    }
    let entry = code_only(&read("crates/ny-cli/src/commands/vnncomp.rs"));
    for needle in [
        "let relational_route = RelationalRouteDecision::for_vnnlib(&vnnlib);",
        "model: resolved_plan.facts.as_ref(),",
        "model_bytes: fs::metadata(&onnx).ok().map(|meta| meta.len()),",
        "facts_reason: Some(\"relational-route\"),",
        "relational_selected: Some(relational_route.selected()),",
        "spec: super::beta_crown::spec_shape::read_spec(&vnnlib),",
    ] {
        assert!(
            entry.contains(needle),
            "the scored router start must still read `{needle}` (else the census reads \
             different facts than the router)"
        );
    }
    let resolver = code_only(&read("crates/ny-cli/src/plan_resolver.rs"));
    assert!(
        fn_body(&resolver, "resolve_and_materialize_with_runtime")
            .contains("let facts = ModelFacts::from_onnx_file(onnx);"),
        "the scored plan's facts are the plan resolver's single-pass scan"
    );
    let glue = code_only(&read("crates/ny-cli/src/router/mod.rs"));
    // #router-r7 split the glue: `policy_input` (the one the census calls) is
    // `policy_input_with(ctx, false)` — the census reads no prior — and the
    // facts and the admission are built in `policy_input_with`.
    assert!(
        fn_body(&glue, "policy_input").contains("policy_input_with(ctx, false)"),
        "the census's policy input must read no prior"
    );
    let pi = fn_body(&glue, "policy_input_with");
    assert!(pi.contains("let facts = facts::instance_facts("));
    assert!(pi.contains("registry::admit(decl, &admission_ctx)"));
    let core = code_only(&read("crates/ny-core/src/router/route.rs"));
    assert!(
        fn_body(&core, "census_decision").contains("commit("),
        "the census decision is CR-RS's commit"
    );
}

/// SOURCE RATCHET: the margin-row probe is the lane's OWN admission. Both
/// entries into the lane still call every function the probe calls, the
/// inline lane's compiler still refuses a net without the head, and the
/// probe calls those same functions.
#[test]
fn census_margin_row_probe_is_the_lanes_own_admission() {
    let lane = code_only(&read("crates/ny-cli/src/commands/margin_row_bab.rs"));
    let inline = fn_body(&lane, "try_margin_row_unsat");
    let concurrent = fn_body(&lane, "spawn_concurrent_lane_impl");
    for (name, body) in [("inline", inline), ("concurrent", concurrent)] {
        for needle in [
            "margin_row_bab_enabled()",
            "extract_twin_spec(onnx)",
            "parse_vnnlib_robustness(vnnlib)",
            "lo.len() != spec.n_in",
        ] {
            assert!(
                body.contains(needle),
                "the {name} entry must check `{needle}`"
            );
        }
    }
    assert!(concurrent.contains("twin_spec_compiles_with_root_policy(&spec, &runtime_policy)"));
    assert!(fn_body(&lane, "twin_spec_compiles_with_root_policy")
        .contains("if !twin_spec_has_margin_row_head(spec) {"));
    assert!(inline.contains("run_margin_row_lane_with_policy("));
    let net = code_only(&read("crates/ny-propagate/src/margin_row/net.rs"));
    let compile = fn_body(&net, "compile_with_root_source");
    assert!(
        compile.contains("expected 2 gemms") && compile.contains("no trunk relus"),
        "the inline lane's compiler refuses a net without the head"
    );
    let model = fn_body(&lane, "lane_model_admission");
    for needle in [
        "margin_row_bab_enabled()",
        "extract_twin_spec(onnx)",
        "twin_spec_has_margin_row_head(&spec)",
    ] {
        assert!(
            model.contains(needle),
            "the model probe must call `{needle}`"
        );
    }
    let property = fn_body(&lane, "lane_property_admission");
    assert!(property.contains("parse_vnnlib_robustness(vnnlib)"));
    assert!(property.contains("lo.len() != n_in"));
}

/// `text` without its `#[cfg(test)]` / `#[cfg(all(test, ..))]` / `#[test]`
/// items (a test module or function). Relies on the tree being formatted (the gate checks
/// `trustfmt`): an item that does not end on its first line closes on the
/// line that is exactly `}` at the attribute's indentation.
fn without_test_items(text: &str) -> String {
    let lines: Vec<&str> = text.lines().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        let line = lines[i];
        let trimmed = line.trim_start();
        if !["#[cfg(test)]", "#[cfg(all(test", "#[test]"]
            .iter()
            .any(|attr| trimmed.starts_with(attr))
        {
            out.push(line);
            i += 1;
            continue;
        }
        let indent = &line[..line.len() - trimmed.len()];
        let rest = trimmed
            .split_once(']')
            .map_or("", |(_, r)| r.trim_start_matches([')', ']']));
        let rest = rest.trim();
        // The item's first line: the attribute's own line when the item
        // follows it there, else the first line after any further attributes.
        let mut j = i;
        if rest.is_empty() {
            j = i + 1;
            while j < lines.len() && lines[j].trim_start().starts_with("#[") {
                j += 1;
            }
        }
        let first = lines.get(j).map_or("", |l| l.trim_end());
        if first.ends_with(';') || first.ends_with('}') {
            i = j + 1;
            continue;
        }
        let close = format!("{indent}}}");
        while j < lines.len() && lines[j].trim_end() != close {
            j += 1;
        }
        i = j + 1;
    }
    out.join("\n")
}

/// Every `.rs` file under `rel` (recursively) that is not a test file, as
/// non-comment, non-test code.
fn non_test_sources(rel: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let mut stack = vec![workspace_root().join(rel)];
    while let Some(dir) = stack.pop() {
        let entries =
            fs::read_dir(&dir).unwrap_or_else(|e| panic!("read_dir {}: {e}", dir.display()));
        for entry in entries {
            let path = entry.expect("dir entry").path();
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if path.is_dir() {
                if name != "tests" {
                    stack.push(path);
                }
            } else if path.extension().is_some_and(|ext| ext == "rs") && !name.ends_with("tests.rs")
            {
                let text = fs::read_to_string(&path).expect("read source");
                let shown = path
                    .strip_prefix(workspace_root())
                    .unwrap_or(&path)
                    .display()
                    .to_string();
                out.push((shown, without_test_items(&code_only(&text))));
            }
        }
    }
    out.sort();
    assert!(
        out.len() > 20,
        "{rel}: only {} source files found: the census would be VOID",
        out.len()
    );
    out
}

/// The name of the fn `trimmed` (a line without its indentation) declares.
fn fn_decl(trimmed: &str) -> Option<String> {
    let mut rest = trimmed;
    for prefix in ["pub(crate) ", "pub(super) ", "pub "] {
        if let Some(r) = rest.strip_prefix(prefix) {
            rest = r;
            break;
        }
    }
    for prefix in ["const ", "async ", "unsafe "] {
        if let Some(r) = rest.strip_prefix(prefix) {
            rest = r;
        }
    }
    rest.strip_prefix("fn ")
        .and_then(|r| r.split(['(', '<']).next())
        .map(str::to_owned)
}

/// The column-0 item a line opens: `Some("mod <name>")` for a module,
/// `Some("")` for any other item, `None` for a line that opens none.
fn column0_item(line: &str) -> Option<String> {
    if line.is_empty() || line.starts_with([' ', '\t', '}', ')', ']', '#']) {
        return None;
    }
    let mut rest = line;
    for prefix in ["pub(crate) ", "pub(super) ", "pub "] {
        if let Some(r) = rest.strip_prefix(prefix) {
            rest = r;
            break;
        }
    }
    Some(rest.strip_prefix("mod ").map_or_else(String::new, |r| {
        format!("mod {}", r.split([' ', '{', ';']).next().unwrap_or(""))
    }))
}

/// `(file, owner)` of every non-test call to any of `calls` under `rel`: the
/// owner is the nearest enclosing fn, prefixed `<module>::` when the call
/// sits inside a column-0 module (a definition `fn <call>` is not a call,
/// nor is a longer name that ends in one).
pub(super) fn call_sites(rel: &str, calls: &[&str]) -> Vec<(String, String)> {
    let mut sites = Vec::new();
    for (file, text) in non_test_sources(rel) {
        let lines: Vec<&str> = text.lines().collect();
        for (k, line) in lines.iter().enumerate() {
            let hit = calls.iter().any(|call| {
                line.match_indices(call).any(|(at, _)| {
                    let before = &line[..at];
                    !before.trim_end().ends_with("fn")
                        && !before.ends_with(|c: char| c.is_ascii_alphanumeric() || c == '_')
                })
            });
            if !hit {
                continue;
            }
            let owner = lines[..=k]
                .iter()
                .rev()
                .find_map(|l| fn_decl(l.trim_start()))
                .unwrap_or_else(|| "<none>".to_owned());
            let module = lines[..=k]
                .iter()
                .rev()
                .find_map(|l| column0_item(l))
                .unwrap_or_default();
            sites.push((
                file.clone(),
                match module.strip_prefix("mod ") {
                    Some(m) => format!("{m}::{owner}"),
                    None => owner,
                },
            ));
        }
    }
    sites
}

/// Review (#admission-census) SOURCE RATCHET: the margin-row lane has
/// EXACTLY the two entries whose checks the probe is built from. The probe
/// may refuse a row only because every entry into the lane makes the probed
/// checks first (`census_margin_row_probe_is_the_lanes_own_admission` pins
/// them in those two entries); a THIRD caller of the lane runner — in the
/// binary or in `ny-propagate` — would reach the consumers on rows the
/// census took out of the moat, so it fails here until the probe (and this
/// list) account for it. Mutation target: a third non-test call site.
#[test]
fn census_margin_row_lane_has_exactly_the_two_probed_entries() {
    let runners = [
        "run_margin_row_lane(",
        "run_margin_row_lane_with_policy(",
        "run_margin_row_lane_with_head(",
    ];
    let lane = "crates/ny-cli/src/commands/margin_row_bab.rs";
    let (research, scored): (Vec<_>, Vec<_>) = call_sites("crates/ny-cli/src", &runners)
        .into_iter()
        .partition(|(file, owner)| file == lane && owner.starts_with("research::"));
    assert_eq!(
        scored,
        vec![
            (lane.to_owned(), "try_margin_row_unsat".to_owned()),
            (lane.to_owned(), "spawn_concurrent_lane_impl".to_owned()),
        ],
        "the binary's non-test calls into the margin-row lane runner (outside the research \
         module) are exactly the two entries the census probe mirrors"
    );
    // The research module's calls run under `ny vnncomp-research margin-row`
    // only — a command the router never starts in, so no route reaches them —
    // and that is the module's one non-test door.
    assert!(
        !research.is_empty(),
        "positive control: the research module's lane calls are seen (else the scan is VOID)"
    );
    let doors: Vec<(String, String)> = non_test_sources("crates/ny-cli/src")
        .into_iter()
        .flat_map(|(file, text)| {
            text.lines()
                .filter(|l| l.contains("margin_row_bab::research::"))
                .map(|l| (file.clone(), l.trim().to_owned()))
                .collect::<Vec<_>>()
        })
        .collect();
    assert_eq!(doors.len(), 1, "the research module's doors: {doors:?}");
    assert!(
        doors[0].1.starts_with("VnncompResearchAction::MarginRow"),
        "the research module is reached only from `ny vnncomp-research margin-row`: {doors:?}"
    );
    // Inside ny-propagate only the runner aliases themselves reach the lane
    // body (no lane, tier or fallback enters it on its own).
    let mut inner = call_sites(
        "crates/ny-propagate/src",
        &[
            "run_margin_row_lane(",
            "run_margin_row_lane_with_policy(",
            "run_margin_row_lane_with_head(",
            "lane_impl(",
            "lane_impl_with_policy(",
        ],
    );
    inner.sort();
    inner.dedup();
    let module = "crates/ny-propagate/src/margin_row/mod.rs";
    assert_eq!(
        inner,
        vec![
            (module.to_owned(), "lane_impl".to_owned()),
            (module.to_owned(), "run_margin_row_lane".to_owned()),
            (
                module.to_owned(),
                "run_margin_row_lane_with_head".to_owned()
            ),
            (
                module.to_owned(),
                "run_margin_row_lane_with_policy".to_owned()
            ),
        ],
        "only the runner aliases call the margin-row lane body inside ny-propagate"
    );
}

/// The margin-row probe admits the twin family on a robustness property and
/// refuses, each by its own reason, a half-space property, a wrong input
/// width, a net outside the compiled head and a net outside the family —
/// and the census hands that answer to the router's own `commit`: the
/// margin-row consumers decline on the refused rows while the router alone
/// would route them, and a capability under another parent is untouched.
#[test]
fn census_margin_row_probe_admits_the_twin_family_only() {
    let world = World::new();
    let twin = world.file("twin.onnx", &twin_model_bytes(2));
    let one_head = world.file("one_head.onnx", &twin_model_bytes(1));
    let mlp = workspace_root().join("tests/models/simple_mlp.onnx");
    let robust = world.file("robust.vnnlib", robustness_property(72).as_bytes());
    let half = world.file("half.vnnlib", halfspace_property(72).as_bytes());
    let narrow = world.file("narrow.vnnlib", robustness_property(71).as_bytes());
    assert_eq!(lane_model_admission(&twin), Ok(72));
    assert_eq!(lane_property_admission(&robust, 72), Ok(()));
    assert_eq!(
        lane_property_admission(&half, 72),
        Err("property-not-robustness")
    );
    assert_eq!(lane_property_admission(&narrow, 72), Err("input-width"));
    assert_eq!(lane_model_admission(&one_head), Err("no-compiled-head"));
    assert_eq!(lane_model_admission(&mlp), Err("model-outside-twin-family"));

    let rows = [
        row("ok", &twin, &robust),
        row("half", &twin, &half),
        row("narrow", &twin, &narrow),
        row("onehead", &one_head, &robust),
    ];
    let (_, cells) = census_cells(&rows, &world.options("probe"));
    let fd = cell(&cells, "ok", "frozen-dual-bab");
    assert_eq!(
        (
            fd[0].as_str(),
            fd[1].as_str(),
            fd[2].as_str(),
            fd[3].as_str()
        ),
        ("1", "1", "admit", "-")
    );
    // #admission-precision: the router now reads the archetype at start, so
    // the half-space row is refused by the margin-row consumers' OWN
    // precondition (robustness only) before any probe is paid for.
    for cap in [
        "frozen-dual-bab",
        "wave-beta-bab",
        "bi-window-joint-refresh",
    ] {
        let c = cell(&cells, "half", cap);
        assert_eq!(
            (c[0].as_str(), c[1].as_str(), c[2].as_str(), c[3].as_str()),
            ("0", "0", "skipped", "admit:decline:spec-not-served"),
            "half/{cap}: the archetype is a fact at start"
        );
    }
    for (id, why) in [("narrow", "input-width"), ("onehead", "no-compiled-head")] {
        for cap in [
            "frozen-dual-bab",
            "wave-beta-bab",
            "bi-window-joint-refresh",
        ] {
            let c = cell(&cells, id, cap);
            assert_eq!(c[0], "0", "{id}/{cap}: the probe's refusal");
            assert_eq!(c[1], "1", "{id}/{cap}: the router alone would route");
            assert_eq!(c[2], format!("decline:{why}"));
            assert_eq!(c[3], format!("parent:margin-row:decline:{why}"));
        }
        // No probe for the input-split parent: the census is the router.
        let cz = cell(&cells, id, "cgan-child-zono");
        assert_eq!((cz[0].as_str(), cz[2].as_str()), (cz[1].as_str(), "-"));
    }
    // The declared M5 host has no device: the device wave declines BEFORE
    // any probe is paid for (`skipped`).
    let cuda = cell(&cells, "ok", "wave-beta-bab-cuda");
    assert_eq!(
        (cuda[0].as_str(), cuda[1].as_str(), cuda[2].as_str()),
        ("0", "0", "skipped")
    );
    assert_eq!(cuda[3], "admit:decline:no-device");
}

/// Review (#admission-census): a probe that CANNOT READ its input keeps the
/// router's deferred answer — the census only ever refuses a row on the
/// lane's own refusal of a readable model and property, never on a failed
/// read. Before the review fix the lane's functions answered `None` for an
/// unreadable property and the census recorded that as the lane's
/// `property-not-robustness` refusal: a row the router routes taken OUT of
/// the moat. Mutation target: an `unknown` probe applied as a decline.
#[test]
fn census_probe_failure_keeps_the_routers_answer() {
    let world = World::new();
    let twin = world.file("twin.onnx", &twin_model_bytes(2));
    let robust = world.file("robust.vnnlib", robustness_property(72).as_bytes());
    // A gzip property that does not decompress (its competition form cannot
    // be built), and a plain property the census process may not read.
    let broken = world.file("broken.vnnlib.gz", b"this is not a gzip stream");
    let locked = world.file("locked.vnnlib", robustness_property(72).as_bytes());
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        fs::set_permissions(&locked, fs::Permissions::from_mode(0o000)).expect("chmod 000");
    }
    // Root (or a non-unix host) reads it anyway: that leg is then not
    // exercised, and says so; the gzip leg always is.
    let locked_enforced = fs::read(&locked).is_err();
    let rows = [
        row("ok", &twin, &robust),
        row("broken", &twin, &broken),
        row("locked", &twin, &locked),
    ];
    let (_, cells) = census_cells(&rows, &world.options("unreadable"));
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        fs::set_permissions(&locked, fs::Permissions::from_mode(0o644)).expect("chmod back");
    }
    let ok = cell(&cells, "ok", "frozen-dual-bab");
    assert_eq!(
        (ok[0].as_str(), ok[1].as_str(), ok[2].as_str()),
        ("1", "1", "admit"),
        "positive control: the readable row is admitted by the probe"
    );
    let mut legs = vec![("broken", "unknown:property-gz-unreadable")];
    if locked_enforced {
        legs.push(("locked", "unknown:property-unreadable"));
    } else {
        eprintln!(
            "census_probe_failure_keeps_the_routers_answer: the locked leg is NOT exercised here"
        );
    }
    for (id, probe) in legs {
        for cap in [
            "frozen-dual-bab",
            "wave-beta-bab",
            "bi-window-joint-refresh",
        ] {
            let c = cell(&cells, id, cap);
            assert_eq!(c[1], "1", "{id}/{cap}: the router alone routes the row");
            assert_eq!(
                c[2], probe,
                "{id}/{cap}: the probe could not read its input"
            );
            assert_eq!(
                (c[0].as_str(), c[3].as_str()),
                ("1", "-"),
                "{id}/{cap}: an unreadable input keeps the router's answer (admitted), never a refusal"
            );
        }
    }
}

/// NAME-BLIND: the same files under category-like and random names, other
/// row ids and the gzip form give identical decisions.
#[test]
fn census_is_name_blind() {
    let world = World::new();
    let bytes = twin_model_bytes(2);
    let robust = robustness_property(72);
    let half = halfspace_property(72);
    let a = [
        row(
            "y26_cifar100_2024_0007",
            &world.file("cifar100_2024/onnx/CIFAR100_resnet_medium.onnx", &bytes),
            &world.file("cifar100_2024/vnnlib/robust_0.vnnlib", robust.as_bytes()),
        ),
        row(
            "y26_cifar100_2024_0008",
            &world.file("cifar100_2024/onnx/CIFAR100_resnet_medium.onnx", &bytes),
            &world.file("cifar100_2024/vnnlib/half_1.vnnlib", half.as_bytes()),
        ),
    ];
    let b = [
        row(
            "q",
            &world.gz("zz/metaroom_2023/x9.onnx.gz", &bytes),
            &world.gz("zz/acasxu/p.vnnlib.gz", robust.as_bytes()),
        ),
        row(
            "r",
            &world.file("tmp/model_final_v3.onnx", &bytes),
            &world.file("tmp/safenlp_prop.vnnlib", half.as_bytes()),
        ),
    ];
    let (_, ca) = census_cells(&a, &world.options("names_a"));
    let (_, cb) = census_cells(&b, &world.options("names_b"));
    assert_eq!(ca.len(), cb.len());
    for ((_, cap_a, fa), (_, cap_b, fb)) in ca.iter().zip(&cb) {
        assert_eq!(cap_a, cap_b);
        assert_eq!(
            &fa[..7],
            &fb[..7],
            "{cap_a}: decisions (or the spec read) differ by name"
        );
    }
    assert!(
        ca.iter().any(|(_, _, f)| f[0] == "1") && ca.iter().any(|(_, _, f)| f[0] == "0"),
        "positive control: the fixture both admits and refuses"
    );
}

/// DETERMINISM + HASH BINDING: the same corpus gives the same bytes; the
/// trailer verifies; any edit, and any change of the corpus or the binary
/// binding, changes the hash.
#[test]
fn census_is_deterministic_and_hash_bound() {
    let world = World::new();
    let model = world.file("net.onnx", &twin_model_bytes(2));
    let property = world.file("prop.vnnlib", robustness_property(72).as_bytes());
    let rows = [row("r1", &model, &property)];
    let options = world.options("det");
    fs::create_dir_all(&options.work_dir).expect("work dir");
    let (one, summary) = census_text(&rows, "c0ffee", "b1", "commit=test", &options);
    let (two, _) = census_text(&rows, "c0ffee", "b1", "commit=test", &options);
    assert_eq!(one, two, "the census is deterministic");
    assert_eq!(
        verify_census_text(&one).expect("verifies"),
        summary.census_sha256
    );
    for (key, value) in [
        ("binary_sha256", "b1"),
        ("corpus_sha256", "c0ffee"),
        ("corpus_rows", "1"),
        ("router_schema", ny_core::router::ROUTER_SCHEMA),
        ("r5_schema", ny_core::router::route::R5_SCHEMA),
        (
            "registry_version",
            super::registry::registry_version().as_str(),
        ),
        ("corpus_schema", CORPUS_SCHEMA),
        ("probes", "margin-row=structural/v1"),
        ("spec_read", "vnnlib-shape/v1 max_bytes=16777216"),
        ("columns", COLUMNS.join("\t").as_str()),
    ] {
        assert!(
            one.lines().any(|l| l == format!("# {key}\t{value}")),
            "binding line `{key}` missing or wrong"
        );
    }
    assert!(one.starts_with(&format!("# {}\n", ny_core::router::route::CENSUS_SCHEMA)));
    assert!(one
        .lines()
        .any(|l| l.starts_with("# envelope\tops=linear+conv+")
            && l.ends_with("specs=robustness+halfspace+disjunctive+relational+nonlinear")));
    assert!(
        !one.contains("other+") && !one.contains("+other"),
        "never `other`"
    );
    let tampered = one.replacen("frozen-dual-bab\t1", "frozen-dual-bab\t0", 1);
    assert_ne!(tampered, one, "positive control: the edit landed");
    assert!(
        verify_census_text(&tampered).is_err(),
        "an edited census is refused"
    );
    let (other_corpus, s2) = census_text(&rows, "c0ffef", "b1", "commit=test", &options);
    assert_ne!(s2.census_sha256, summary.census_sha256);
    assert!(verify_census_text(&other_corpus).is_ok());
    let (_, s3) = census_text(&rows, "c0ffee", "b2", "commit=test", &options);
    assert_ne!(
        s3.census_sha256, summary.census_sha256,
        "the binary is bound"
    );
    // The command refuses to overwrite a census.
    fs::write(
        &options.manifest,
        format!(
            "# {CORPUS_SCHEMA}\nr1\t{}\t{}\t100\n",
            model.display(),
            property.display()
        ),
    )
    .expect("manifest");
    fs::write(&options.out, "existing").expect("existing out");
    assert!(census::handle_census_command(&options).is_err());
}

/// The corpus manifest and the envelope flag fail closed.
#[test]
fn census_corpus_and_envelope_parse_fail_closed() {
    let good = format!("# {CORPUS_SCHEMA}\n# a comment\nr1\t/a.onnx\t/a.vnnlib\t116.0\n\nr2\t/b.onnx\t/b.vnnlib\t20\n");
    let rows = parse_corpus(&good).expect("parses");
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0].budget_secs, 116);
    for bad in [
        "r1\t/a.onnx\t/a.vnnlib\t1\n".to_owned(),
        format!("# {CORPUS_SCHEMA}\nr1\t/a.onnx\t/a.vnnlib\n"),
        format!("# {CORPUS_SCHEMA}\nr1\t/a\t/b\t1\nr1\t/c\t/d\t1\n"),
        format!("# {CORPUS_SCHEMA}\nr1\t/a\t/b\t0\n"),
        format!("# {CORPUS_SCHEMA}\n"),
    ] {
        assert!(parse_corpus(&bad).is_err(), "{bad:?}");
    }
    assert_eq!(
        parse_envelope("widest").expect("widest"),
        CENSUS_WIDEST_ENVELOPE
    );
    let e = parse_envelope("conv+relu+linear/robustness").expect("tokens");
    assert_eq!(
        e.ops,
        OpSet::EMPTY
            .with(OpKind::Conv)
            .with(OpKind::Relu)
            .with(OpKind::Linear)
    );
    assert_eq!(
        e.specs,
        SpecSet::EMPTY.with(ny_core::router::decl::SpecArchetype::Robustness)
    );
    for bad in ["conv", "conv+magic/robustness", "conv/vibes"] {
        assert!(parse_envelope(bad).is_err(), "{bad}");
    }
}

/// A row the entry point refuses before the router starts has no route; a
/// relational property takes the relational route, whose router has no
/// facts, so every capability is refused there.
#[test]
fn census_without_a_router_or_facts_declines() {
    let world = World::new();
    let model = world.file("net.onnx", &twin_model_bytes(2));
    let property = world.file("prop.vnnlib", robustness_property(72).as_bytes());
    let relational = world.file("rel.vnnlib", RELATIONAL.as_bytes());
    let rows = [
        row("missing", &world.dir.path().join("absent.onnx"), &property),
        row("rel", &model, &relational),
        row("noprop", &model, &world.dir.path().join("absent.vnnlib")),
    ];
    let (_, cells) = census_cells(&rows, &world.options("none"));
    for (r, _, f) in &cells {
        assert_eq!(f[0], "0", "{r}: {f:?}");
        assert_eq!(f[1], "0", "{r}: {f:?}");
    }
    assert_eq!(
        cell(&cells, "missing", "frozen-dual-bab")[3],
        "no-router:model-missing"
    );
    assert_eq!(
        cell(&cells, "noprop", "frozen-dual-bab")[3],
        "no-router:property-missing"
    );
    let rel = cell(&cells, "rel", "frozen-dual-bab");
    assert_eq!(rel[4], "unknown:relational-route");
    assert_eq!(rel[3], "admit:unknown:facts-unknown");
}

/// #prepared-root-route: the census decides `prepared-root` by CR-RS's own
/// checks with its OWN predicate asked on the census lever context (a row
/// with no category file, zero NY_*): admitted on a robustness row of the
/// twin family, refused on a half-space row (its call site serves argmax
/// robustness only), never probed (it has no parent). And the census input
/// honours the predicate: started with a lever set that arms a dose the
/// route refuses, the same row is refused `own:decline:<the lane's reason>`.
#[test]
fn census_prepared_root_follows_its_own_predicate() {
    let world = World::new();
    let model = world.file("net.onnx", &twin_model_bytes(2));
    let robust = world.file("robust.vnnlib", robustness_property(72).as_bytes());
    let half = world.file("half.vnnlib", halfspace_property(72).as_bytes());
    let rows = [row("rob", &model, &robust), row("half", &model, &half)];
    let (_, cells) = census_cells(&rows, &world.options("prepared"));
    let rob = cell(&cells, "rob", "prepared-root");
    assert_eq!(
        (
            rob[0].as_str(),
            rob[1].as_str(),
            rob[2].as_str(),
            rob[3].as_str()
        ),
        ("1", "1", "-", "-"),
        "{rob:?}"
    );
    let half_row = cell(&cells, "half", "prepared-root");
    assert_eq!(half_row[0], "0", "{half_row:?}");
    assert_eq!(half_row[3], "admit:decline:spec-not-served");
    // The census input honours the predicate (positive control).
    let scan = ModelFacts::from_onnx_file(&model).expect("the scan reads the fixture");
    let frozen_dual = &ny_levers::decls::margin_row::MARGIN_ROW_FROZEN_DUAL_BAB;
    let armed = ny_levers::LeverSet::freeze_with(ny_levers::all(), |key| {
        (key == frozen_dual.name).then(|| "1".to_owned())
    });
    let defaults = census::no_category_levers();
    let decide = |levers: &ny_levers::LeverSet| {
        let input = policy_input(&StartContext {
            levers: Some(levers),
            authority: Authority::Frozen,
            model: Some(&scan),
            model_bytes: fs::metadata(&model).ok().map(|m| m.len()),
            facts_reason: None,
            declared_networks: Some(0),
            relational_selected: Some(false),
            official_budget_secs: 100,
            internal_secs: 100,
            join_margin: crate::commands::vnncomp::CONCURRENT_LANE_JOIN_MARGIN,
            anchor: Instant::now(),
            target_host: Some(HostClass::M5),
            pinned_keys: Vec::new(),
            trial_pinned_keys: Vec::new(),
            spec: crate::commands::beta_crown::spec_shape::read_spec(&robust),
        });
        ny_core::router::route::census_decision(&input, "prepared-root", CENSUS_WIDEST_ENVELOPE)
            .expect("routable")
    };
    assert!(decide(&defaults).routed);
    let refused = decide(&armed);
    assert!(!refused.routed);
    assert_eq!(
        refused.reasons,
        ["own:decline:unsupported-armed-consumer-dose"]
    );
}
