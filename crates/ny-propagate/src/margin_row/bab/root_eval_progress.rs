// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Stage timing under NY_MARGIN_ROW_ROOT_TIMING. The legacy evaluator has no
//! deadline parameter; observations do not establish proof or deadline checks.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

struct ActiveEvaluation {
    id: u64,
    started: Instant,
    finished: bool,
}

pub(super) struct Evaluation(Option<ActiveEvaluation>);

fn emit(id: u64, event: &str, details: std::fmt::Arguments<'_>) {
    use std::io::Write;
    let mut stderr = std::io::stderr().lock();
    let _ = writeln!(
        stderr,
        "[head-progress] v=1 pid={} evaluation={id} event={event} {details}",
        std::process::id()
    );
    let _ = stderr.flush();
}

impl Evaluation {
    pub(super) fn new(inputs: usize, head: usize, classes: usize) -> Self {
        if std::env::var("NY_MARGIN_ROW_ROOT_TIMING").is_err() {
            return Self(None);
        }
        static NEXT: AtomicU64 = AtomicU64::new(1);
        let active = ActiveEvaluation {
            id: NEXT.fetch_add(1, Ordering::Relaxed),
            started: Instant::now(),
            finished: false,
        };
        emit(
            active.id,
            "eval_begin",
            format_args!("inputs={inputs} head={head} classes={classes} deadline=unavailable"),
        );
        Self(Some(active))
    }

    pub(super) fn stage(&self, name: &'static str, columns: usize) -> Option<Stage> {
        let active = self.0.as_ref()?;
        let stage = Stage {
            id: active.id,
            name,
            columns,
            evaluation_started: active.started,
            started: Instant::now(),
            finished: false,
        };
        stage.emit("stage_begin");
        Some(stage)
    }

    pub(super) fn finish(&mut self, remaining: usize) {
        if let Some(active) = &mut self.0 {
            active.finished = true;
            emit(
                active.id,
                "eval_complete",
                format_args!(
                    "remaining_classes={remaining} seconds={:.6}",
                    active.started.elapsed().as_secs_f64()
                ),
            );
        }
    }
}

impl Drop for Evaluation {
    fn drop(&mut self) {
        if let Some(active) = &self.0 {
            if !active.finished {
                emit(
                    active.id,
                    "eval_abort",
                    format_args!("seconds={:.6}", active.started.elapsed().as_secs_f64()),
                );
            }
        }
    }
}

pub(super) struct Stage {
    id: u64,
    name: &'static str,
    columns: usize,
    evaluation_started: Instant,
    started: Instant,
    finished: bool,
}

impl Stage {
    fn emit(&self, event: &str) {
        emit(
            self.id,
            event,
            format_args!(
                "stage={} columns={} seconds={:.6} evaluation_seconds={:.6}",
                self.name,
                self.columns,
                self.started.elapsed().as_secs_f64(),
                self.evaluation_started.elapsed().as_secs_f64()
            ),
        );
    }

    pub(super) fn finish(stage: Option<Self>) {
        if let Some(mut stage) = stage {
            stage.finished = true;
            stage.emit("stage_end");
        }
    }
}

impl Drop for Stage {
    fn drop(&mut self) {
        if !self.finished {
            self.emit("stage_abort");
        }
    }
}
