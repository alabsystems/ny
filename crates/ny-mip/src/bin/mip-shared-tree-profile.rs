// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

#![forbid(unsafe_code)]

//! Offline profiler for a captured NY phase-split family.

use std::path::{Path, PathBuf};

use ny_mip::dump::from_milp_text;
use ny_mip::shared_tree_profile::profile_phase_split_family;

fn usage() -> ! {
    eprintln!(
        "usage: mip-shared-tree-profile --root <root.milp> --children <dir-or-child.milp> \
         [<child.milp> ...]"
    );
    std::process::exit(2);
}

fn main() {
    if let Err(error) = run() {
        eprintln!("mip-shared-tree-profile: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let mut args = std::env::args_os().skip(1);
    if args.next().as_deref() != Some(std::ffi::OsStr::new("--root")) {
        usage();
    }
    let root_path = args.next().map(PathBuf::from).unwrap_or_else(|| usage());
    if args.next().as_deref() != Some(std::ffi::OsStr::new("--children")) {
        usage();
    }
    let child_inputs: Vec<PathBuf> = args.map(PathBuf::from).collect();
    if child_inputs.is_empty() {
        usage();
    }

    let mut child_paths = Vec::new();
    for input in child_inputs {
        if input.is_dir() {
            let mut entries: Vec<PathBuf> = std::fs::read_dir(&input)
                .map_err(|error| format!("read {}: {error}", input.display()))?
                .filter_map(|entry| entry.ok().map(|entry| entry.path()))
                .filter(|path| path.extension().and_then(|ext| ext.to_str()) == Some("milp"))
                .collect();
            entries.sort();
            child_paths.extend(entries);
        } else {
            child_paths.push(input);
        }
    }
    if child_paths.is_empty() {
        return Err("no child .milp files found".to_string());
    }

    let root = load(&root_path)?;
    let children = child_paths
        .iter()
        .map(|path| load(path))
        .collect::<Result<Vec<_>, _>>()?;
    let profile = profile_phase_split_family(&root, &children)
        .map_err(|error| format!("family rejected: {error}"))?;
    let json = serde_json::to_string_pretty(&profile)
        .map_err(|error| format!("serialize profile: {error}"))?;
    println!("{json}");
    Ok(())
}

fn load(path: &Path) -> Result<ny_mip::ir::MilpProblem, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|error| format!("read {}: {error}", path.display()))?;
    from_milp_text(&text).map_err(|error| format!("parse {}: {error}", path.display()))
}
