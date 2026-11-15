// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Evidence job (#forward-linear-cost-gate): how far past its deadline an
//! over-admitted forward-linear build stops.
//!
//! The default-lane test `adv_over_admission_is_bounded_by_mid_build_deadline_abort`
//! pins the semantics: the cost gate, lied to by the forward-linear MAC-rate
//! override, admits a build of hundreds of GMACs, and the build's own deadline
//! polls stop it with a typed `DeadlineExceeded`. The overshoot past the
//! deadline is a wall-clock quantity: the length of the unpolled stretch the
//! deadline happens to fall in, stretched by host load (TEST_CONFORMANCE.md,
//! "Measurements and exhaustive audits"). This example measures it and fails
//! when it exceeds `--max-overshoot-secs` (default 30).
//!
//! It must run with the rate override exported to an absurd rate, exactly as
//! TEST_CONFORMANCE.md shows, in a debug build (an optimized build may finish
//! the fixture before its deadline), under `scripts/ny_box.sh measure`. Exit
//! codes: 0 within the bound, 1 over it, 2 VOID (the gate did not admit on the
//! override, the build finished, or it failed some other way), 64 usage.

use std::process::ExitCode;
use std::time::{Duration, Instant};

use ndarray::{Array1, ArrayD, IxDyn};
use ny_core::NyError;
use ny_propagate::layers::Conv2dLayer;
use ny_propagate::{forward_linear_admission_record, GraphNetwork, GraphNode, Layer};
use ny_tensor::BoundedTensor;

/// Just above the cold-build headroom floor (30 s), so the gate's affordability
/// arithmetic, not the floor, decides admission.
const DEADLINE_SECS: u64 = 32;

/// The default-lane test's fixture: 6 convs x 32 channels over 32x32, about
/// 296 G f64 MACs, with coefficient state (32768 x 3072) under the dense cap.
fn fixture() -> (GraphNetwork, BoundedTensor) {
    let mut graph = GraphNetwork::new();
    let mut prev: Option<String> = None;
    for i in 0..6 {
        let name = format!("conv{i}");
        let in_c = if i == 0 { 3 } else { 32 };
        let kernel_len = 32 * in_c * 9;
        let kernel = ArrayD::from_shape_vec(
            IxDyn(&[32, in_c, 3, 3]),
            (0..kernel_len)
                .map(|j| ((j % 5) as f32 - 2.0) / 50.0)
                .collect(),
        )
        .expect("kernel shape");
        let conv =
            Conv2dLayer::with_input_shape(kernel, Some(Array1::zeros(32)), (1, 1), (1, 1), 32, 32)
                .expect("valid conv");
        let node = match prev {
            None => GraphNode::from_input(&name, Layer::Conv2d(conv)),
            Some(p) => GraphNode::new(&name, Layer::Conv2d(conv), vec![p]),
        };
        graph.add_node(node);
        prev = Some(name);
    }
    graph.set_output(&prev.expect("at least one conv"));
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[3, 32, 32]), -0.01f32),
        ArrayD::from_elem(IxDyn(&[3, 32, 32]), 0.01f32),
    )
    .expect("input bounds");
    (graph, input)
}

fn max_overshoot_secs() -> Option<f64> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.as_slice() {
        [] => Some(30.0),
        [flag, value] if flag == "--max-overshoot-secs" => value
            .parse::<f64>()
            .ok()
            .filter(|v| v.is_finite() && *v >= 0.0),
        _ => None,
    }
}

fn main() -> ExitCode {
    let Some(bound) = max_overshoot_secs() else {
        eprintln!("usage: forward_linear_over_admission_abort [--max-overshoot-secs SECONDS]");
        return ExitCode::from(64);
    };
    let (graph, input) = fixture();
    let started = Instant::now();
    let deadline = started + Duration::from_secs(DEADLINE_SECS);
    let outcome = graph.collect_forward_linear_bounds_dag_cached(&input, None, Some(deadline));
    let wall = started.elapsed();
    // This process runs one build, so the most recent admission record is its.
    let record = forward_linear_admission_record();
    let admitted_on_override = record
        .as_ref()
        .is_some_and(|r| r.admitted && r.rate_source == "env");
    let message = match outcome {
        Err(NyError::DeadlineExceeded(message)) => message,
        Ok(_) => {
            println!(
                "FL_OVER_ADMISSION VOID reason=build-finished wall_s={:.3} (an optimized build \
                 can finish the fixture; measure a debug build)",
                wall.as_secs_f64()
            );
            return ExitCode::from(2);
        }
        Err(other) => {
            println!("FL_OVER_ADMISSION VOID reason=other-error error={other:?}");
            return ExitCode::from(2);
        }
    };
    if !admitted_on_override || !message.contains("forward-linear") {
        println!(
            "FL_OVER_ADMISSION VOID reason=not-admitted-on-override record={record:?} \
             message={message} (export the forward-linear MAC-rate override as \
             TEST_CONFORMANCE.md shows)"
        );
        return ExitCode::from(2);
    }
    let overshoot = wall.saturating_sub(Duration::from_secs(DEADLINE_SECS));
    let within = overshoot.as_secs_f64() <= bound;
    println!(
        "FL_OVER_ADMISSION {} wall_s={:.3} overshoot_s={:.3} bound_s={bound} message={message}",
        if within { "OK" } else { "OVER" },
        wall.as_secs_f64(),
        overshoot.as_secs_f64(),
    );
    if within {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    }
}
