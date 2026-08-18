// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Blast-radius harness: attempt `load_onnx` on every `.onnx` under a root and
//! print one TSV row per model.
//!
//! Usage: `model_load_sweep <root> [<root> ...]`
//! Output: `<status>\t<path>\t<detail>` where status is OK or FAIL, and detail
//! is the layer count (OK) or the single-line refusal message (FAIL).
//!
//! Run it at the base commit and at the change, then diff: any model whose
//! status flips is in the blast radius, in either direction.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

fn collect(root: &Path, out: &mut Vec<PathBuf>) -> std::io::Result<()> {
    let entries = std::fs::read_dir(root)?;
    let mut children = Vec::new();
    for entry in entries {
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

fn main() -> ExitCode {
    let roots: Vec<_> = std::env::args_os().skip(1).map(PathBuf::from).collect();
    if roots.is_empty() {
        eprintln!("usage: model_load_sweep <root> [<root> ...]");
        return ExitCode::FAILURE;
    }

    let mut models = Vec::new();
    for root in &roots {
        if let Err(error) = collect(root, &mut models) {
            eprintln!("cannot scan {}: {error}", root.display());
            return ExitCode::FAILURE;
        }
    }
    if models.is_empty() {
        eprintln!("no .onnx models found under the requested roots");
        return ExitCode::FAILURE;
    }
    eprintln!("scanning {} models", models.len());
    let mut failures = 0usize;
    for model in models {
        // A panic inside one model must not abort the sweep.
        let path = model.clone();
        let result = std::panic::catch_unwind(move || ny_onnx::load_onnx(&path));
        match result {
            Ok(Ok(loaded)) => println!("OK\t{}\t{}", model.display(), loaded.network.layers.len()),
            Ok(Err(error)) => {
                failures += 1;
                println!(
                    "FAIL\t{}\t{}",
                    model.display(),
                    error.to_string().replace(['\n', '\t'], " ")
                );
            }
            Err(_) => {
                failures += 1;
                println!("PANIC\t{}\t-", model.display());
            }
        }
    }
    if failures == 0 {
        ExitCode::SUCCESS
    } else {
        eprintln!("model-load sweep found {failures} failure(s)");
        ExitCode::FAILURE
    }
}
