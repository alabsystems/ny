// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use std::fs;
use std::path::{Path, PathBuf};

use ny_propagate::{BetaCrownConfig, ConvMode};

use super::{apply_preset, load_preset};

fn collect_yaml_files(dir: &Path, output: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(dir)
        .unwrap_or_else(|err| panic!("failed to read config directory {}: {err}", dir.display()))
    {
        let entry = entry.unwrap_or_else(|err| {
            panic!(
                "failed to inspect config entry under {}: {err}",
                dir.display()
            )
        });
        let path = entry.path();
        if path.is_dir() {
            collect_yaml_files(&path, output);
        } else if path
            .extension()
            .is_some_and(|extension| extension == "yaml")
        {
            output.push(path);
        }
    }
}

#[test]
fn every_vnncomp_preset_keeps_quarantined_cut_authority_dark() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let configs_dir = repo_root.join("configs");
    let mut yaml_files = Vec::new();

    for entry in fs::read_dir(&configs_dir).expect("configs directory should be readable") {
        let entry = entry.expect("config directory entry should be readable");
        let path = entry.path();
        let is_vnncomp_dir = path.is_dir()
            && entry
                .file_name()
                .to_str()
                .is_some_and(|name| name.starts_with("vnncomp"));
        if is_vnncomp_dir {
            collect_yaml_files(&path, &mut yaml_files);
        }
    }
    yaml_files.sort();
    assert!(
        !yaml_files.is_empty(),
        "expected at least one VNN-COMP preset to audit"
    );

    for path in yaml_files {
        let preset = load_preset(&path)
            .unwrap_or_else(|err| panic!("VNN-COMP preset {} must parse: {err}", path.display()));
        assert_ne!(
            preset.bab.cuts.enabled,
            Some(true),
            "{} enables quarantined cut proof authority",
            path.display()
        );
        assert_ne!(
            preset.bab.cuts.near_miss,
            Some(true),
            "{} enables unproved near-miss cuts",
            path.display()
        );
        assert_ne!(
            preset.bab.cuts.proactive,
            Some(true),
            "{} enables unproved proactive cuts",
            path.display()
        );
    }
}

#[test]
fn relusplitter_scored_presets_are_dark_but_keep_matrix_throughput() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let paths = [
        "configs/vnncomp25/relusplitter.yaml",
        "configs/vnncomp25/relusplitter_rsplitter_gpu_bab.yaml",
        "configs/vnncomp25/relusplitter_rsplitter_multiobjective_wgpu.yaml",
    ];

    for relative_path in paths {
        let preset =
            load_preset(&repo_root.join(relative_path)).expect("relusplitter preset should load");
        assert_eq!(preset.bab.cuts.enabled, Some(false), "{relative_path}");
        assert_eq!(preset.bab.cuts.near_miss, Some(false), "{relative_path}");
        assert_eq!(preset.bab.cuts.proactive, Some(false), "{relative_path}");
        assert_eq!(
            preset.general.conv_mode,
            Some(ConvMode::Matrix),
            "{relative_path} must retain the measured dense-matrix Conv2d lane"
        );

        let mut config = BetaCrownConfig::default();
        apply_preset(&mut config, &preset)
            .unwrap_or_else(|err| panic!("failed to apply {relative_path}: {err}"));
        config
            .validate()
            .unwrap_or_else(|err| panic!("{relative_path} must validate: {err}"));
        assert!(!config.enable_cuts, "{relative_path}");
        assert!(!config.enable_near_miss_cuts, "{relative_path}");
        assert!(!config.enable_proactive_cuts, "{relative_path}");
        assert!(
            !config.use_patches(),
            "{relative_path} must remain in matrix mode"
        );
    }
}
