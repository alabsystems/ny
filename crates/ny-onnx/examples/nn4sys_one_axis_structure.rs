// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Source-only NN4SYS post-loader one-axis algebra census.
//!
//! This example is not linked into `ny`, is never called by a verifier, and
//! computes no bounds or verdict.  It loads an ONNX model through NY's actual
//! `GraphNetwork` conversion, prints the post-loader operation census, and runs
//! the hard-capped structural recognizer for explicitly supplied free axes.
//!
//! Example:
//!
//! ```text
//! cargo run -p ny-onnx --example nn4sys_one_axis_structure -- \
//!   /absolute/path/mscn_2048d.onnx 54,68,82
//! ```
//!
//! Resolve benchmark symlinks before invoking this tool: the hardened ONNX
//! loader intentionally rejects a model path that escapes its retained parent
//! directory through a symlink.

use std::collections::BTreeMap;
use std::error::Error;
use std::io;
use std::time::{Duration, Instant};

use ny_onnx::load_onnx;
use ny_propagate::{GraphNetwork, OneAxisAlgebraReport};

const MAX_AXES: usize = 64;
const PER_AXIS_DEADLINE: Duration = Duration::from_secs(2);

type Result<T> = std::result::Result<T, Box<dyn Error>>;

fn invalid_input(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

fn parse_axes(raw: &str) -> Result<Vec<usize>> {
    let axes: Vec<usize> = raw
        .split(',')
        .map(str::trim)
        .map(|value| {
            if value.is_empty() || !value.bytes().all(|byte| byte.is_ascii_digit()) {
                return Err(invalid_input(
                    "free axes must be comma-separated unsigned decimals",
                ));
            }
            value
                .parse::<usize>()
                .map_err(|error| invalid_input(format!("invalid free axis {value:?}: {error}")))
        })
        .collect::<std::result::Result<_, _>>()?;
    if axes.is_empty() || axes.len() > MAX_AXES {
        return Err(invalid_input(format!("supply between 1 and {MAX_AXES} free axes")).into());
    }
    Ok(axes)
}

fn op_census(graph: &GraphNetwork) -> Result<BTreeMap<&'static str, usize>> {
    let mut census = BTreeMap::new();
    for name in graph.exec_order()? {
        let node = graph
            .node(name)
            .ok_or_else(|| invalid_input(format!("execution-order node {name:?} is missing")))?;
        *census.entry(node.layer().layer_type()).or_insert(0) += 1;
    }
    Ok(census)
}

fn render_report(report: &OneAxisAlgebraReport) {
    let class = report
        .class
        .map(|class| format!("{class:?}"))
        .unwrap_or_else(|| "DECLINED".to_string());
    let decline = report
        .decline
        .as_ref()
        .map(|decline| {
            format!(
                "{:?}@{}",
                decline.reason,
                decline.node.as_deref().unwrap_or("-")
            )
        })
        .unwrap_or_else(|| "-".to_string());
    println!(
        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
        report.free_axis,
        class,
        decline,
        report.nodes_examined,
        report.dynamic_relu_nodes,
        report.constant_sided_mul_nodes,
        report.constant_divisor_nodes.len(),
        report.dynamic_sigmoid_nodes,
        report.static_sigmoid_nodes,
    );
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let model_path = args
        .next()
        .ok_or_else(|| invalid_input("usage: nn4sys_one_axis_structure MODEL AXIS[,AXIS...]"))?;
    let axes = parse_axes(
        &args
            .next()
            .ok_or_else(|| invalid_input("missing free-axis list"))?,
    )?;
    if args.next().is_some() {
        return Err(invalid_input("unexpected extra arguments").into());
    }

    let model = load_onnx(&model_path)?;
    let [input] = model.network.inputs.as_slice() else {
        return Err(invalid_input("expected exactly one ONNX graph input").into());
    };
    if input.shape.iter().any(|&dimension| dimension <= 0) {
        return Err(invalid_input("ONNX input shape must be fully static and positive").into());
    }
    let input_shape = if input.shape.len() > 1 {
        if input.shape[0] != 1 {
            return Err(invalid_input("expected a singleton ONNX batch dimension").into());
        }
        &input.shape[1..]
    } else {
        input.shape.as_slice()
    }
    .iter()
    .map(|&dimension| {
        usize::try_from(dimension)
            .map_err(|_| invalid_input("ONNX input dimension does not fit usize"))
    })
    .collect::<std::result::Result<Vec<_>, _>>()?;
    let graph = model.to_graph_network()?;
    println!(
        "model={model_path}\tnodes={}\tinput_shape={input_shape:?}\toutput={}",
        graph.num_nodes(),
        graph.output_name()
    );
    println!("post_loader_ops={:?}", op_census(&graph)?);
    println!(
        "axis\tclass\tdecline\tnodes\tdynamic_relus\tconstant_sided_muls\t\
         constant_divisor_obligations\tdynamic_sigmoids\tstatic_sigmoids"
    );
    for axis in axes {
        let deadline = Instant::now()
            .checked_add(PER_AXIS_DEADLINE)
            .unwrap_or_else(Instant::now);
        let report = graph.recognize_one_free_axis_algebra_until(&input_shape, axis, deadline);
        render_report(&report);
    }
    Ok(())
}
