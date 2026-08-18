// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Source-only exact one-axis phase-certificate probe for canonical NN4SYS
//! VNN-LIB clauses.
//!
//! This example is not linked into `ny`, has no production caller, and prints
//! `verdict_authority=false`.  It parses the selected canonical one-line
//! conjunction with exact decimal rationals, generates a phase record, then
//! independently replays that record.
//!
//! ```text
//! cargo run -p ny-onnx --example nn4sys_one_axis_phase -- \
//!   MODEL.onnx PROPERTY.vnnlib CLAUSE_INDEX
//! ```

use std::error::Error;
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::time::{Duration, Instant};

use ny_onnx::load_onnx;
use ny_propagate::{
    Layer, OneAxisConstraintRelation, OneAxisExactProblem, OneAxisOutputConstraint,
    OneAxisPhaseLimits, OneAxisRational,
};

const MAX_PROPERTY_BYTES: u64 = 128 << 20;
const MAX_LINE_BYTES: usize = 1 << 20;
// Exact `BigRational` graph arithmetic is intentionally expensive in debug
// builds.  This is a source-only probe, so give generation plus independent
// replay a bounded five-minute wall budget while retaining fail-closed expiry.
const PHASE_DEADLINE: Duration = Duration::from_mins(5);

type Result<T> = std::result::Result<T, Box<dyn Error>>;

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

fn tokens(line: &str) -> Vec<&str> {
    let mut result = Vec::new();
    let mut start = None;
    for (index, character) in line.char_indices() {
        if character == '(' || character == ')' || character.is_ascii_whitespace() {
            if let Some(begin) = start.take() {
                result.push(&line[begin..index]);
            }
            if character == '(' || character == ')' {
                result.push(&line[index..=index]);
            }
        } else if start.is_none() {
            start = Some(index);
        }
    }
    if let Some(begin) = start {
        result.push(&line[begin..]);
    }
    result
}

fn variable_index(name: &str, prefix: &str) -> Option<usize> {
    name.strip_prefix(prefix)?
        .bytes()
        .all(|byte| byte.is_ascii_digit())
        .then(|| name[prefix.len()..].parse().ok())
        .flatten()
}

fn selected_clause(path: &str, clause_index: usize) -> Result<String> {
    let file = File::open(path)?;
    let metadata = file.metadata()?;
    if metadata.len() > MAX_PROPERTY_BYTES {
        return Err(invalid("property exceeds the 128 MiB source-probe cap").into());
    }
    let mut selected = None;
    let mut seen = 0usize;
    let mut total = 0u64;
    for line in BufReader::new(file).lines() {
        let line = line?;
        total = total
            .checked_add(u64::try_from(line.len())?)
            .ok_or_else(|| invalid("property byte count overflow"))?;
        if total > MAX_PROPERTY_BYTES || line.len() > MAX_LINE_BYTES {
            return Err(invalid("property or canonical clause line exceeds its cap").into());
        }
        if line.trim_start().starts_with("(and ") {
            if seen == clause_index {
                selected = Some(line);
            }
            seen = seen
                .checked_add(1)
                .ok_or_else(|| invalid("clause count overflow"))?;
        }
    }
    selected.ok_or_else(|| {
        invalid(format!(
            "clause index {clause_index} is out of range ({seen} canonical clauses)"
        ))
        .into()
    })
}

fn parse_problem(line: &str, input_shape: Vec<usize>) -> Result<OneAxisExactProblem> {
    let input_elements = input_shape
        .iter()
        .try_fold(1usize, |product, &dimension| product.checked_mul(dimension))
        .ok_or_else(|| invalid("input shape product overflow"))?;
    let tokens = tokens(line);
    if tokens.len() < 3 || tokens[0] != "(" || tokens[1] != "and" {
        return Err(invalid("selected clause is not a canonical one-line conjunction").into());
    }
    let mut lower = vec![None; input_elements];
    let mut upper = vec![None; input_elements];
    let mut constraints = Vec::new();
    let mut cursor = 2usize;
    while tokens.get(cursor) == Some(&"(") {
        let [Some(&operation), Some(&variable), Some(&literal), Some(&close)] = [
            tokens.get(cursor + 1),
            tokens.get(cursor + 2),
            tokens.get(cursor + 3),
            tokens.get(cursor + 4),
        ] else {
            return Err(invalid("truncated canonical atom").into());
        };
        if close != ")" || !matches!(operation, "<=" | ">=") {
            return Err(invalid("unsupported canonical atom").into());
        }
        let value = OneAxisRational::parse_decimal(literal)
            .ok_or_else(|| invalid(format!("invalid exact decimal {literal:?}")))?;
        if let Some(index) = variable_index(variable, "X_") {
            if index >= input_elements {
                return Err(invalid(format!("input variable {variable:?} is out of range")).into());
            }
            let target = if operation == ">=" {
                &mut lower[index]
            } else {
                &mut upper[index]
            };
            if target.replace(value).is_some() {
                return Err(invalid(format!("duplicate {operation} bound for {variable}")).into());
            }
        } else if variable == "Y_0" {
            constraints.push(OneAxisOutputConstraint {
                relation: if operation == "<=" {
                    OneAxisConstraintRelation::LessEqual
                } else {
                    OneAxisConstraintRelation::GreaterEqual
                },
                bound: value,
            });
        } else {
            return Err(invalid(format!("unsupported variable {variable:?}")).into());
        }
        cursor += 5;
    }
    if tokens.get(cursor) != Some(&")") || cursor + 1 != tokens.len() {
        return Err(invalid("trailing or malformed canonical clause tokens").into());
    }
    if constraints.is_empty() {
        return Err(invalid("selected clause has no scalar Y_0 constraint").into());
    }
    let lower = lower
        .into_iter()
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| invalid("selected clause does not bound every input below"))?;
    let upper = upper
        .into_iter()
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| invalid("selected clause does not bound every input above"))?;
    let free_axes = lower
        .iter()
        .zip(&upper)
        .enumerate()
        .filter_map(|(index, (left, right))| (left != right).then_some(index))
        .collect::<Vec<_>>();
    let [free_axis] = free_axes.as_slice() else {
        return Err(invalid(format!(
            "selected clause has {} free axes, not exactly one",
            free_axes.len()
        ))
        .into());
    };
    Ok(OneAxisExactProblem {
        input_shape,
        fixed_inputs: lower.clone(),
        free_axis: *free_axis,
        lower: lower[*free_axis].clone(),
        upper: upper[*free_axis].clone(),
        constraints,
    })
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let model_path = args
        .next()
        .ok_or_else(|| invalid("usage: nn4sys_one_axis_phase MODEL VNNLIB CLAUSE_INDEX"))?;
    let property_path = args.next().ok_or_else(|| invalid("missing VNN-LIB path"))?;
    let clause_index = args
        .next()
        .ok_or_else(|| invalid("missing clause index"))?
        .parse::<usize>()
        .map_err(|error| invalid(format!("invalid clause index: {error}")))?;
    if args.next().is_some() {
        return Err(invalid("unexpected extra arguments").into());
    }

    let model = load_onnx(&model_path)?;
    let [input] = model.network.inputs.as_slice() else {
        return Err(invalid("expected exactly one ONNX graph input").into());
    };
    if input.shape.iter().any(|&dimension| dimension <= 0) {
        return Err(invalid("ONNX input shape must be fully static and positive").into());
    }
    let dimensions = if input.shape.len() > 1 {
        if input.shape[0] != 1 {
            return Err(invalid("expected singleton ONNX batch dimension").into());
        }
        &input.shape[1..]
    } else {
        input.shape.as_slice()
    };
    let input_shape = dimensions
        .iter()
        .map(|&dimension| {
            usize::try_from(dimension).map_err(|_| invalid("input dimension does not fit usize"))
        })
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let clause = selected_clause(&property_path, clause_index)?;
    let problem = parse_problem(&clause, input_shape)?;
    let graph = model.to_graph_network()?;
    if std::env::var_os("NY_ONE_AXIS_DUMP_GRAPH").is_some() {
        for name in graph.exec_order()? {
            let node = graph
                .node(name)
                .ok_or_else(|| invalid(format!("missing graph node {name:?}")))?;
            if let Layer::Linear(linear) = node.layer() {
                eprintln!(
                    "linear={name}\tin={}\tout={}\tshape={:?}\tinputs={:?}",
                    linear.in_features(),
                    linear.out_features(),
                    graph.declared_shape(name),
                    node.inputs(),
                );
            }
        }
    }
    let deadline = Instant::now()
        .checked_add(PHASE_DEADLINE)
        .unwrap_or_else(Instant::now);
    let attempt = graph.exact_one_axis_phase_certificate_until(
        &problem,
        OneAxisPhaseLimits::default(),
        deadline,
    );
    let Some(certificate) = attempt.certificate else {
        println!(
            "model={model_path}\tproperty={property_path}\tclause={clause_index}\t\
             free_axis={}\tverdict_authority=false\tcells_examined={}\t\
             exact_operations={}\tdecline={:?}",
            problem.free_axis,
            attempt.phase_cells_examined,
            attempt.exact_operations,
            attempt.decline
        );
        return Ok(());
    };
    let replay = graph.replay_exact_one_axis_phase_certificate_until(
        &problem,
        &certificate,
        OneAxisPhaseLimits::default(),
        deadline,
    );
    println!(
        "model={model_path}\tproperty={property_path}\tclause={clause_index}\t\
         free_axis={}\tverdict_authority={}\tcells={}\tobservation={:?}\t\
         exact_operations={}\treplay_accepted={}\treplay_decline={:?}",
        problem.free_axis,
        certificate.verdict_authority,
        certificate.cells.len(),
        certificate.observation,
        attempt.exact_operations,
        replay.accepted,
        replay.decline,
    );
    Ok(())
}
