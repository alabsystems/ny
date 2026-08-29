// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Corpus-wide OpaqueSkip census.
//!
//! For every `.onnx` under the roots given on the command line, reproduce the
//! COMPETITION load path (`load_onnx_with_config` with subprocess ORT shape
//! inference + `GraphNetworkOptions { compound_node_policy:
//! DecomposeNormalization, .. }`, exactly as
//! `crates/ny-cli/src/commands/beta_crown/model_load.rs`), then:
//!
//!   * collect every `warn!` the loader/builder emitted for this model, and
//!   * recompute the IBP taint set the SAME way `graph_ibp.rs` does (seed on
//!     `Layer::OpaqueSkip`, forward-close over `exec_order`), and report
//!     whether the graph's OUTPUT node lands in it.
//!
//! Output-tainted is the fatal case: `graph_ibp.rs` returns fully unbounded
//! output bounds ("sound unknown") and no verdict can ever be derived.
//! A skipped node that does NOT reach the output may be harmless.
//!
//! Usage: `opaqueskip_census <ny-exe-for-shape-infer> <root> [<root> ...]`
//! Output: TSV on stdout, one row per model:
//!   status  path  nodes  skips  output_tainted  tainted_nodes  skip_ops  reasons

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::{Arc, Mutex};

use ny_onnx::{
    load_onnx_with_config, CompoundNodePolicy, GraphNetworkOptions, OnnxLoadConfig,
    ShapeInferBackend,
};
use ny_propagate::Layer;

/// Sink that a `tracing` fmt subscriber writes every event into, so the
/// loader's own `warn!` text (the REASON a node was skipped) is captured
/// verbatim instead of re-derived.
#[derive(Clone, Default)]
struct Sink(Arc<Mutex<Vec<u8>>>);

impl Sink {
    fn take(&self) -> String {
        let mut guard = self.0.lock().expect("sink poisoned");
        let text = String::from_utf8_lossy(&guard).into_owned();
        guard.clear();
        text
    }
}

impl Write for Sink {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().expect("sink poisoned").extend_from_slice(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for Sink {
    type Writer = Sink;
    fn make_writer(&'a self) -> Self::Writer {
        self.clone()
    }
}

fn collect(root: &Path, out: &mut Vec<PathBuf>) -> std::io::Result<()> {
    let mut children = Vec::new();
    for entry in std::fs::read_dir(root)? {
        children.push(entry?.path());
    }
    children.sort();
    for child in children {
        if child.is_dir() {
            collect(&child, out)?;
        } else if child.extension().is_some_and(|ext| ext == "onnx") {
            out.push(child);
        }
    }
    Ok(())
}

fn one_line(text: &str) -> String {
    text.replace(['\n', '\t', '\r'], " ")
}

fn main() -> ExitCode {
    let mut args: Vec<PathBuf> = std::env::args_os().skip(1).map(PathBuf::from).collect();
    if args.len() < 2 {
        eprintln!("usage: opaqueskip_census <ny-exe> <root> [<root> ...]");
        return ExitCode::FAILURE;
    }
    let ny_exe = args.remove(0);

    let sink = Sink::default();
    tracing_subscriber::fmt()
        .with_writer(sink.clone())
        .with_ansi(false)
        .with_target(false)
        .with_max_level(tracing::Level::WARN)
        .init();

    let mut models = Vec::new();
    for root in &args {
        if let Err(error) = collect(root, &mut models) {
            eprintln!("cannot scan {}: {error}", root.display());
            return ExitCode::FAILURE;
        }
    }
    eprintln!("scanning {} models", models.len());

    let config =
        OnnxLoadConfig::default().with_shape_infer_backend(ShapeInferBackend::Subprocess {
            exe: ny_exe,
        });
    let options = GraphNetworkOptions {
        compound_node_policy: CompoundNodePolicy::DecomposeNormalization,
        ..GraphNetworkOptions::default()
    };

    println!("status\tpath\tnodes\tskips\toutput_tainted\ttainted_nodes\tskip_ops\treasons");
    for model in &models {
        let _ = sink.take();
        let path = model.clone();
        let cfg = config.clone();
        let loaded = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
            load_onnx_with_config(&path, &cfg)
        }));
        let onnx = match loaded {
            Ok(Ok(m)) => m,
            Ok(Err(error)) => {
                println!(
                    "LOAD_FAIL\t{}\t-\t-\t-\t-\t-\t{}",
                    model.display(),
                    one_line(&error.to_string())
                );
                continue;
            }
            Err(_) => {
                println!("LOAD_PANIC\t{}\t-\t-\t-\t-\t-\t-", model.display());
                continue;
            }
        };

        let built = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            onnx.to_graph_network_with_options(options)
        }));
        let graph = match built {
            Ok(Ok(g)) => g,
            Ok(Err(error)) => {
                println!(
                    "GRAPH_FAIL\t{}\t-\t-\t-\t-\t-\t{}",
                    model.display(),
                    one_line(&error.to_string())
                );
                continue;
            }
            Err(_) => {
                println!("GRAPH_PANIC\t{}\t-\t-\t-\t-\t-\t-", model.display());
                continue;
            }
        };

        let logs = sink.take();

        // Recompute graph_ibp.rs's taint set: seed on OpaqueSkip, forward-close
        // over the execution order. Same predicate, same order.
        let exec_order: Vec<String> = match graph.exec_order() {
            Ok(order) => order.to_vec(),
            Err(error) => {
                println!(
                    "EXEC_ORDER_FAIL\t{}\t{}\t-\t-\t-\t-\t{}",
                    model.display(),
                    graph.num_nodes(),
                    one_line(&error.to_string())
                );
                continue;
            }
        };
        let mut tainted: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut skips: Vec<String> = Vec::new();
        for name in &exec_order {
            let Some(node) = graph.node(name) else {
                continue;
            };
            let is_skip = matches!(node.layer(), Layer::OpaqueSkip(_));
            if is_skip {
                skips.push(name.clone());
            }
            if is_skip || node.inputs().iter().any(|i| tainted.contains(i.as_str())) {
                tainted.insert(name.clone());
            }
        }
        let output = graph.output_name().to_string();
        let effective_output = if output.is_empty() {
            exec_order.last().cloned().unwrap_or_default()
        } else {
            output
        };
        let output_tainted = tainted.contains(&effective_output);

        // The reasons: every warn line the loader emitted that names a skip.
        let reasons: Vec<String> = logs
            .lines()
            .filter(|line| {
                line.contains("Skipping unsupported layer")
                    || line.contains("Skipping Reshape")
                    || line.contains("conservative unbounded bounds")
                    || line.contains("conservative bounds")
            })
            .map(one_line)
            .collect();

        println!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            if skips.is_empty() { "CLEAN" } else { "SKIP" },
            model.display(),
            graph.num_nodes(),
            skips.len(),
            output_tainted,
            tainted.len(),
            skips.join(" | "),
            reasons.join(" || ")
        );
        let _ = std::io::stdout().flush();
    }
    ExitCode::SUCCESS
}
