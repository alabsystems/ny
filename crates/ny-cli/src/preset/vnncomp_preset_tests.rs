// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! VNN-COMP benchmark preset loading tests.
//!
//! Extracted from `tests.rs` to stay under the 1000-line limit.
//! These tests load actual YAML configs from `configs/vnncomp25/` and verify
//! that parsed preset values flow correctly to `BetaCrownConfig`.

use super::apply::apply_preset;
use super::*;
use crate::commands::beta_crown::branching::{
    apply_resolved_auto_branching_runtime_policy, resolve_auto_branching, AutoBranchingReason,
    AutoBranchingRequest, ModelStructure,
};
use ny_propagate::{
    BetaCrownConfig, BranchingHeuristic, InputClipType, PgdAlphaMode, PgdOptimizer,
};
use std::{fs, path::Path};

#[test]
fn no_shipped_preset_arms_solver_dark_bound_paths() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs");
    let mut configured = Vec::new();

    for year in ["vnncomp24", "vnncomp25", "vnncomp26"] {
        let dir = root.join(year);
        let Ok(entries) = fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries {
            let path = entry.expect("preset directory entry").path();
            if path.extension().and_then(|extension| extension.to_str()) != Some("yaml") {
                continue;
            }
            let text = fs::read_to_string(&path).expect("readable shipped preset");
            let preset: PresetConfig = serde_yaml::from_str(&text)
                .unwrap_or_else(|error| panic!("preset {} must parse: {error}", path.display()));
            if preset.solver.const_matmul_interval.is_some()
                || preset.solver.joint_bilinear_quad.is_some()
            {
                configured.push(path);
            }
        }
    }

    assert!(
        configured.is_empty(),
        "the const-matmul and joint-bilinear bound paths must remain absent from every shipped \
         competition preset: {configured:?}"
    );
}

#[test]
fn only_vnncomp25_cifar100_arms_root_alpha_phase_checkpoint() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs");
    let mut armed = Vec::new();

    for year in ["vnncomp24", "vnncomp25", "vnncomp26"] {
        let dir = root.join(year);
        let Ok(entries) = fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries {
            let path = entry.expect("preset directory entry").path();
            if path.extension().and_then(|extension| extension.to_str()) != Some("yaml") {
                continue;
            }
            let text = fs::read_to_string(&path).expect("readable shipped preset");
            let preset: PresetConfig = serde_yaml::from_str(&text)
                .unwrap_or_else(|error| panic!("preset {} must parse: {error}", path.display()));
            if preset.bab.root_alpha_phase_checkpoint == Some(true) {
                armed.push(
                    path.strip_prefix(&root)
                        .expect("preset below config root")
                        .to_path_buf(),
                );
            }
        }
    }
    armed.sort();
    assert_eq!(
        armed,
        [Path::new("vnncomp25/cifar100_2024.yaml").to_path_buf()],
        "the root-alpha checkpoint production opt-in must remain scoped to measured CIFAR100"
    );
}

#[test]
fn effective_cifar100_2024_preset_arms_root_alpha_phase_checkpoint() {
    let configs_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs");
    let effective = crate::commands::vnncomp::resolve_preset_path(&configs_root, "cifar100_2024")
        .expect("CIFAR100 production preset resolves");
    let preset = load_preset(&effective).expect("effective CIFAR100 preset loads");
    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).expect("effective CIFAR100 preset applies");
    assert!(
        config.root_alpha_phase_checkpoint,
        "newer vnncomp directories must not silently shadow the qualified CIFAR100 policy"
    );
}

#[test]
fn only_vnncomp25_cifar100_arms_kfsb_cert_reuse() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs");
    let mut armed = Vec::new();

    for year in ["vnncomp24", "vnncomp25", "vnncomp26"] {
        let dir = root.join(year);
        let Ok(entries) = fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries {
            let path = entry.expect("preset directory entry").path();
            if path.extension().and_then(|extension| extension.to_str()) != Some("yaml") {
                continue;
            }
            let text = fs::read_to_string(&path).expect("readable shipped preset");
            let preset: PresetConfig = serde_yaml::from_str(&text)
                .unwrap_or_else(|error| panic!("preset {} must parse: {error}", path.display()));
            if preset.bab.branching.kfsb_cert_reuse == Some(true) {
                armed.push(
                    path.strip_prefix(&root)
                        .expect("preset below config root")
                        .to_path_buf(),
                );
            }
        }
    }
    armed.sort();
    assert_eq!(
        armed,
        [Path::new("vnncomp25/cifar100_2024.yaml").to_path_buf()],
        "the proof-carrying kFSB reuse opt-in must remain scoped to qualified CIFAR100"
    );
}

#[test]
fn effective_cifar100_2024_preset_arms_kfsb_cert_reuse() {
    let configs_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs");
    let effective = crate::commands::vnncomp::resolve_preset_path(&configs_root, "cifar100_2024")
        .expect("CIFAR100 production preset resolves");
    assert!(
        effective.ends_with("vnncomp25/cifar100_2024.yaml"),
        "the missing vnncomp26 override must fall back to the vnncomp25 CIFAR100 preset: {}",
        effective.display()
    );
    let preset = load_preset(&effective).expect("effective CIFAR100 preset loads");
    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).expect("effective CIFAR100 preset applies");
    assert!(
        config.kfsb_cert_reuse,
        "newer vnncomp directories must not silently shadow the CIFAR100 certificate policy"
    );
}

#[test]
fn only_vnncomp25_acas_arms_relational_exact_gradient_wrapper() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs");
    let mut armed = Vec::new();

    for year in ["vnncomp24", "vnncomp25", "vnncomp26"] {
        let dir = root.join(year);
        let Ok(entries) = fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries {
            let path = entry.expect("preset directory entry").path();
            if path.extension().and_then(|extension| extension.to_str()) != Some("yaml") {
                continue;
            }
            let text = fs::read_to_string(&path).expect("readable shipped preset");
            let preset: PresetConfig = serde_yaml::from_str(&text)
                .unwrap_or_else(|error| panic!("preset {} must parse: {error}", path.display()));
            if preset.attack.vnncomp_upfront_relational_exact_gradient == Some(true) {
                armed.push(
                    path.strip_prefix(&root)
                        .expect("preset below config root")
                        .to_path_buf(),
                );
            }
        }
    }
    armed.sort();
    assert_eq!(
        armed,
        [Path::new("vnncomp25/acasxu_2023.yaml").to_path_buf()],
        "the default-off relational wrapper opt-in must remain ACAS-only"
    );
}

#[test]
fn no_shipped_preset_arms_mo_cuda_factory_engine_handoff() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs");
    let mut armed = Vec::new();

    for year in ["vnncomp24", "vnncomp25", "vnncomp26"] {
        let dir = root.join(year);
        let Ok(entries) = fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries {
            let path = entry.expect("preset directory entry").path();
            if path.extension().and_then(|extension| extension.to_str()) != Some("yaml") {
                continue;
            }
            let text = fs::read_to_string(&path).expect("readable shipped preset");
            let preset: PresetConfig = serde_yaml::from_str(&text)
                .unwrap_or_else(|error| panic!("preset {} must parse: {error}", path.display()));
            if preset.bab.mo_cuda_factory_engine_handoff.is_some() {
                armed.push(path);
            }
        }
    }

    assert!(
        armed.is_empty(),
        "the default-dark handoff must remain absent from every shipped preset: {armed:?}"
    );
}

#[test]
fn no_shipped_preset_arms_mo_cuda_bounded_shared_executor() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs");
    let mut armed = Vec::new();

    for year in ["vnncomp24", "vnncomp25", "vnncomp26"] {
        let dir = root.join(year);
        let Ok(entries) = fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries {
            let path = entry.expect("preset directory entry").path();
            if path.extension().and_then(|extension| extension.to_str()) != Some("yaml") {
                continue;
            }
            let text = fs::read_to_string(&path).expect("readable shipped preset");
            let preset: PresetConfig = serde_yaml::from_str(&text)
                .unwrap_or_else(|error| panic!("preset {} must parse: {error}", path.display()));
            if preset.bab.mo_cuda_bounded_shared_executor.is_some() {
                armed.push(path);
            }
        }
    }

    assert!(
        armed.is_empty(),
        "the default-dark bounded shared executor must remain absent from every shipped preset: \
         {armed:?}"
    );
}

#[test]
fn cifar100_critical_alpha_bracket_uses_beta_crown_lr_not_adaptive_default() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let cifar = load_preset(&repo_root.join("configs/vnncomp25/cifar100_2024.yaml")).unwrap();
    assert_eq!(cifar.bab.beta_crown.lr_alpha, Some(0.1));

    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &cifar).unwrap();
    assert_eq!(
        config.alpha_lr.to_bits(),
        0.1_f32.to_bits(),
        "the bracket base must be the applied beta-CROWN alpha LR"
    );
    assert_eq!(
        config.adaptive_config.alpha_lr.to_bits(),
        0.01_f32.to_bits(),
        "the sealed adaptive default remains distinct and must not seed the bracket"
    );
    assert_eq!(
        [0.3_f32, 1.0, 2.0].map(|scale| (config.alpha_lr * scale).to_bits()),
        [0.1_f32 * 0.3, 0.1, 0.2].map(f32::to_bits),
        "the applied CIFAR base yields the staged 0.03/0.10/0.20 bracket"
    );
}

#[test]
fn cifar100_2024_arms_bounded_dense_head_crown_only_in_its_preset() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let cifar = load_preset(&repo_root.join("configs/vnncomp25/cifar100_2024.yaml")).unwrap();
    assert_eq!(
        cifar.general.device.as_deref(),
        Some("wgpu"),
        "the CIFAR preset must pass a WGPU engine into the ordinary verifier route"
    );
    assert_eq!(cifar.bab.root_crown_interm_dense_head, Some(true));
    assert_eq!(cifar.bab.root_crown_interm_max_secs, Some(2));
    assert_eq!(cifar.bab.root_crown_interm_max_dim, Some(512));

    let mut cifar_config = BetaCrownConfig::default();
    apply_preset(&mut cifar_config, &cifar).unwrap();
    assert!(cifar_config.root_crown_interm_dense_head);
    assert_eq!(cifar_config.root_crown_interm_max_secs, 2);
    assert_eq!(cifar_config.root_crown_interm_max_dim, 512);
    assert_eq!(
        cifar_config.alpha_config.iterations, 20,
        "CIFAR keeps its alpha-update schedule until a fast initial-CROWN route \
         preserves the useful state without regressing banked UNSAT rows"
    );

    let acas = load_preset(&repo_root.join("configs/vnncomp25/acasxu_2023.yaml")).unwrap();
    assert_eq!(acas.bab.root_crown_interm_dense_head, None);
    let mut acas_config = BetaCrownConfig::default();
    apply_preset(&mut acas_config, &acas).unwrap();
    assert!(
        !acas_config.root_crown_interm_dense_head,
        "representative non-cifar presets must remain default-off"
    );
}

#[test]
fn cifar100_2024_types_the_exact_open_row_adaptive_route() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let cifar = load_preset(&repo_root.join("configs/vnncomp25/cifar100_2024.yaml")).unwrap();
    assert_eq!(cifar.margin_row.adaptive_reserve, Some(true));

    let acas = load_preset(&repo_root.join("configs/vnncomp25/acasxu_2023.yaml")).unwrap();
    assert_eq!(
        acas.margin_row.adaptive_reserve, None,
        "unrelated categories retain the historical reserve policy"
    );
}

#[test]
fn cifar100_2024_enables_reference_adaptive_relu_split_depth() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let cifar = load_preset(&repo_root.join("configs/vnncomp25/cifar100_2024.yaml")).unwrap();
    assert_eq!(cifar.bab.min_batch_size_ratio, Some(0.1));
    assert_eq!(cifar.bab.max_split_depth, Some(4));

    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &cifar).unwrap();
    assert_eq!(config.min_batch_fill_ratio, 0.1);
    assert_eq!(config.max_relu_split_depth, 4);
    assert_eq!(
        config.effective_relu_split_depth(1),
        4,
        "singleton CIFAR root should commit the reference 16-leaf split"
    );
}

/// #a2-parity-not-exercised (measured 2026-08-09, GB10, `--features mip,cuda`).
///
/// The asymmetry pinned below is NOT an oversight waiting to be corrected by a
/// copy from the sibling preset. Arming the whole cifar100 bundle on
/// tinyimagenet — `batch_size` 128 -> 256, `candidates` 10 -> 7, the `clip`
/// block, `kfsb_multi`, `max_split_depth`, `min_batch_size_ratio`,
/// `alpha_crown.lr_alpha` 0.2 -> 0.25, `beta_crown.iterations` 15 -> 10 and
/// `root_crown_interm_dense_head` — was A/B'd at the official 100 s budget,
/// arms interleaved per instance, on both halves of the field (the GT-decided
/// rows ny misses and the rows it banks).
///
/// The arms were confirmed LIVE from the run's own effective-config receipt,
/// and the result was still null, because the internal verifier reports
/// `domains_explored: 0` on this category: every key in the bundle except the
/// dense head is consumed inside the BaB loop, which never runs. The verdicts
/// come from the concurrent margin-row lane, whose entry point
/// (`ny_propagate::margin_row::run_margin_row_lane`) takes no `BetaCrownConfig`
/// at all, so no `bab.*` key can reach it by signature.
///
/// See `docs/TINYIMAGENET_PARITY_A2_NOT_EXERCISED_2026-08-09.md`. Re-attempt
/// only after `domains_explored > 0` is demonstrated here.
#[test]
fn complete_clip_experiment_is_scoped_to_cifar100_not_tinyimagenet() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let cifar = load_preset(&repo_root.join("configs/vnncomp25/cifar100_2024.yaml")).unwrap();
    let tiny = load_preset(&repo_root.join("configs/vnncomp25/tinyimagenet_2024.yaml")).unwrap();

    assert_eq!(cifar.bab.clip.interm_domain, Some(true));
    assert_eq!(cifar.bab.clip.interm_topk, Some(20));
    assert_eq!(cifar.bab.max_split_depth, Some(4));
    assert_eq!(cifar.bab.branching.kfsb_multi, Some(true));
    assert_eq!(cifar.bab.branching.kfsb_cert_reuse, Some(true));
    assert_eq!(cifar.bab.root_alpha_phase_checkpoint, Some(true));

    let mut cifar_config = BetaCrownConfig::default();
    apply_preset(&mut cifar_config, &cifar).unwrap();
    assert!(cifar_config.enable_clip_interm_domain);
    assert_eq!(cifar_config.clip_interm_topk, 20);
    assert_eq!(cifar_config.max_relu_split_depth, 4);
    assert!(cifar_config.use_kfsb_multi_branching);
    assert!(cifar_config.kfsb_cert_reuse);
    assert!(cifar_config.root_alpha_phase_checkpoint);

    assert_eq!(tiny.bab.clip.interm_domain, None);
    assert_eq!(tiny.bab.clip.interm_topk, None);
    assert_eq!(tiny.bab.max_split_depth, None);
    assert_eq!(tiny.bab.branching.kfsb_multi, None);
    assert_eq!(tiny.bab.branching.kfsb_cert_reuse, None);
    assert_eq!(tiny.bab.root_alpha_phase_checkpoint, None);

    let mut tiny_config = BetaCrownConfig::default();
    apply_preset(&mut tiny_config, &tiny).unwrap();
    assert!(!tiny_config.enable_clip_interm_domain);
    assert_eq!(tiny_config.clip_interm_topk, 3);
    assert_eq!(tiny_config.max_relu_split_depth, 1);
    assert!(!tiny_config.use_kfsb_multi_branching);
    assert!(!tiny_config.kfsb_cert_reuse);
    assert!(!tiny_config.root_alpha_phase_checkpoint);
}

/// #margin-row-branch-width: pin the measured cifar100 width and its scoping.
///
/// k=16 converts `idx_8600_sidx_2721` (timeout -> unsat, 186 expansions,
/// frontier drains 424 -> 2) and k=4 LOSES a banked proof, so this key is
/// evidence-bearing in both directions: the value matters, and it must not
/// leak to categories where it was never measured (the tinyimagenet transfer
/// trap all over again — see the allocation-trio no-go below).
#[test]
fn cifar100_branch_width_is_pinned_and_scoped() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let cifar = load_preset(&repo_root.join("configs/vnncomp25/cifar100_2024.yaml")).unwrap();
    assert_eq!(cifar.margin_row.k_head, Some(16));
    assert_eq!(cifar.margin_row.k_trunk, Some(16));

    let tiny = load_preset(&repo_root.join("configs/vnncomp25/tinyimagenet_2024.yaml")).unwrap();
    assert_eq!(tiny.margin_row.k_head, None);
    assert_eq!(tiny.margin_row.k_trunk, None);

    // #backward-interm ships to BOTH wall categories — each on its own
    // measured evidence and its own pre-registered ship gate (2026-08-19).
    assert_eq!(cifar.margin_row.backward_interm, Some(true));
    assert_eq!(tiny.margin_row.backward_interm, Some(true));
}

/// Names of every `margin_row` field this preset actually sets.
///
/// Field-wise rather than a derived comparison on purpose: `MarginRowPreset`
/// gains keys faster than presets adopt them, and a NEW key must show up here
/// as a compile error in [`margin_row_wall_preset_field_diff_is_deliberate`]
/// rather than slip through an `..` rest pattern unexamined.
fn margin_row_fields_set(m: &MarginRowPreset) -> Vec<&'static str> {
    let MarginRowPreset {
        reserve_secs,
        adaptive_reserve,
        reserve_max_frac,
        release_frac,
        concurrent,
        root_f32,
        tableau_f32,
        lru,
        parallel,
        k_head,
        k_trunk,
        k_adaptive,
        gpu,
        gpu_batch,
        gpu_backward,
        alpha_opt,
        alpha_iters,
        alpha_secs,
        alpha_lr,
        alpha_cols,
        alpha_guard,
        clip,
        clip_interm,
        clip_rows,
        clip_topk,
        closure_thresh,
        closure_warmup,
        closure_window,
        closure_max_tier,
        bi_chunk,
        bi_chunk_secs,
        bi_topk,
        beta,
        beta_eta,
        beta_iters,
        beta_per_column,
        beta_columns,
        column_retire,
        parallel_prebuild,
        beta_lambda,
        beta_polyak,
        beta_heads,
        rebake,
        rebake_rounds,
        rebake_secs,
        closure_adaptive,
        branch_reduce_max,
        backward_interm,
        domain_alpha,
        domain_alpha_iters,
        domain_alpha_eta,
        domain_alpha_min_depth,
        domain_alpha_stride,
        domain_alpha_gap,
        bi_secs,
        epoch_tier0,
        epoch_universe,
        epoch_retain,
        epoch_depth,
        epoch_attempts,
        epoch_parent_topk,
        early_exit,
        verifier_threads,
        concurrent_threads,
    } = m;
    let flags: [(&'static str, bool); 64] = [
        ("reserve_secs", reserve_secs.is_some()),
        ("adaptive_reserve", adaptive_reserve.is_some()),
        ("reserve_max_frac", reserve_max_frac.is_some()),
        ("release_frac", release_frac.is_some()),
        ("concurrent", concurrent.is_some()),
        ("root_f32", root_f32.is_some()),
        ("tableau_f32", tableau_f32.is_some()),
        ("lru", lru.is_some()),
        ("parallel", parallel.is_some()),
        ("k_head", k_head.is_some()),
        ("k_trunk", k_trunk.is_some()),
        ("k_adaptive", k_adaptive.is_some()),
        ("gpu", gpu.is_some()),
        ("gpu_batch", gpu_batch.is_some()),
        ("gpu_backward", gpu_backward.is_some()),
        ("alpha_opt", alpha_opt.is_some()),
        ("alpha_iters", alpha_iters.is_some()),
        ("alpha_secs", alpha_secs.is_some()),
        ("alpha_lr", alpha_lr.is_some()),
        ("alpha_cols", alpha_cols.is_some()),
        ("alpha_guard", alpha_guard.is_some()),
        ("clip", clip.is_some()),
        ("clip_interm", clip_interm.is_some()),
        ("clip_rows", clip_rows.is_some()),
        ("clip_topk", clip_topk.is_some()),
        ("closure_thresh", closure_thresh.is_some()),
        ("closure_warmup", closure_warmup.is_some()),
        ("closure_window", closure_window.is_some()),
        ("closure_max_tier", closure_max_tier.is_some()),
        ("bi_chunk", bi_chunk.is_some()),
        ("bi_chunk_secs", bi_chunk_secs.is_some()),
        ("bi_topk", bi_topk.is_some()),
        ("beta", beta.is_some()),
        ("beta_eta", beta_eta.is_some()),
        ("beta_iters", beta_iters.is_some()),
        ("beta_per_column", beta_per_column.is_some()),
        ("beta_columns", beta_columns.is_some()),
        ("column_retire", column_retire.is_some()),
        ("parallel_prebuild", parallel_prebuild.is_some()),
        ("beta_lambda", beta_lambda.is_some()),
        ("beta_polyak", beta_polyak.is_some()),
        ("beta_heads", beta_heads.is_some()),
        ("rebake", rebake.is_some()),
        ("rebake_rounds", rebake_rounds.is_some()),
        ("rebake_secs", rebake_secs.is_some()),
        ("closure_adaptive", closure_adaptive.is_some()),
        ("branch_reduce_max", branch_reduce_max.is_some()),
        ("backward_interm", backward_interm.is_some()),
        ("domain_alpha", domain_alpha.is_some()),
        ("domain_alpha_iters", domain_alpha_iters.is_some()),
        ("domain_alpha_eta", domain_alpha_eta.is_some()),
        ("domain_alpha_min_depth", domain_alpha_min_depth.is_some()),
        ("domain_alpha_stride", domain_alpha_stride.is_some()),
        ("domain_alpha_gap", domain_alpha_gap.is_some()),
        ("bi_secs", bi_secs.is_some()),
        ("epoch_tier0", epoch_tier0.is_some()),
        ("epoch_universe", epoch_universe.is_some()),
        ("epoch_retain", epoch_retain.is_some()),
        ("epoch_depth", epoch_depth.is_some()),
        ("epoch_attempts", epoch_attempts.is_some()),
        ("epoch_parent_topk", epoch_parent_topk.is_some()),
        ("early_exit", early_exit.is_some()),
        ("verifier_threads", verifier_threads.is_some()),
        ("concurrent_threads", concurrent_threads.is_some()),
    ];
    flags
        .into_iter()
        .filter_map(|(name, set)| set.then_some(name))
        .collect()
}

/// The margin-row lane is the prover for BOTH wall categories, and the campaign
/// has repeatedly shipped a cifar100 key and then discovered nobody asked
/// whether tinyimagenet wanted it. This test is the field-by-field diff of the
/// two `margin_row:` sections in executable form, so a transfer is a deliberate
/// edit here rather than a silent copy there.
///
/// The decisions, each with the evidence recorded next to the key in the yaml:
///
/// | field | cifar100 | tinyimagenet | why they differ |
/// |---|---|---|---|
/// | `reserve_secs` | absent (45 default, then released) | `0` | see `adaptive_reserve` |
/// | `adaptive_reserve` | `true` | absent | both hand the verifier the full budget; the ONLY difference is that the adaptive route additionally arms the scored sparse root CROWN (`enables_scored_sparse_crown`), and this category's internal verifier reports `domains_explored: 0` |
/// | `concurrent` / `root_f32` | `true` | `true` | same twin-wall pair, measured separately on each |
/// | `tableau_f32` | absent | `true` | ARMED ON TINY'S OWN EVIDENCE 2026-08-25: conv 41.4->24.6 s, idx_5451 timeout->unsat 3/3, and the 11-row wall-hugger ship gate held 3/3. NOT transferred to cifar100, which sets root_f32 but has no A/B of its own (alloc-trio rule) |
/// | `lru` / `parallel` | absent (64 / all workers) | absent (64 / all workers) | typed delivery makes both measured throughput defaults searchable without silently arming a new YAML treatment; explicit parallel `0` is the serial control and typed `1` is invalid because legacy `1` means all |
/// | `k_head` / `k_trunk` | `16` | absent (8) | cifar's evidence is a frontier explosion; this lane runs a 2-expansion tree |
/// | `k_adaptive` | absent | absent | typed delivery exists; measured 3/3 on cifar, but it remains dark and unmeasured on tiny |
/// | `alpha_opt` | `true` | absent | cifar100 retains its own timeout-to-UNSAT conversion and 9/9 moat; Tiny stays dark because the same pre-tree wall displaced needed tree work |
/// | `gpu*`; explicit alpha tuning; `clip*` / beta / rebake fields | absent | absent | typed search delivery only; cifar100 consumes the alpha defaults under `alpha_opt`, but no explicit tuning or other treatment is qualified to ship without its own sealed A/B |
/// | `closure_adaptive` | absent | absent | #closure-adaptive ships dark pending its own measurement; arming it on one category is exactly the transfer this test exists to make deliberate |
/// | `backward_interm` | `true` | `true` | shipped to each on its own ship gate (2026-08-19) |
/// | `domain_alpha` (+ `_iters`, `_eta`, and the SCHEDULE `_min_depth` / `_stride` / `_gap`) | absent | absent | #margin-row-domain-alpha ships dark pending its pre-registered 2x2 (off / clip / alpha / BOTH) on idx_1298 and idx_4597 — and the schedule grid is MEASURED NULL (2026-08-24: zero conversions at every operating point; see margin_row/domain_alpha.rs); arming it on one category is exactly the transfer this test exists to make deliberate |
/// | `bi_secs` | `40` (measured) | ABSENT | the old "2.16x overrun" rationale confused root-build `secs=` with phase `work_secs=`. The later Tiny dose response solved idx_5185 only 1/4 and did not bank; idx_9611 failed under every dose on that dirty binary, so it is not a causal loss. Tableau-f32 later changed the timing frame again. Tiny stays at 20 pending a clean current-binary 20-vs-30 receipt-closed A/B; see docs/BI_DOSE_RESPONSE_5185_2026-08-24.md |
/// | `bi_chunk_secs` | ABSENT | ABSENT | #bi-adaptive-chunk ships DARK (default 0 = the historical fixed grain). The exploratory idx_9611 observation used unretained logs and a dirty binary, and its declaration cited an unrelated commit; it is context, not scored provenance. Any non-zero value remains an unmeasured experiment requiring a sealed execution receipt |
/// | `epoch_tier0` / `epoch_universe` / `epoch_retain` | absent | absent | typed delivery makes the ranker measurable from the clean scorecard, but promotion does not arm the July ranker result on any shipped category |
/// | `epoch_depth` / `epoch_attempts` / `epoch_parent_topk` | absent | absent | typed delivery exists so the clean scorecard can measure it, but the authoritative refresh remains dark pending a sealed A/B |
/// | `early_exit` | absent (= ON) | absent (= ON) | default; the key exists only as a kill switch |
/// | `verifier_threads` | absent | absent | Front B's unmeasured scheduling seam |
/// | `concurrent_threads` | absent | absent | unmeasured lane-width seam; omission uses the shared global pool without a verifier request, or derives only from observed host-minus-actual-global width with one; explicit `0` forces shared global |
#[test]
fn margin_row_wall_preset_field_diff_is_deliberate() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let dir = repo_root.join("configs/vnncomp25");
    let cifar = load_preset(&dir.join("cifar100_2024.yaml")).unwrap();
    let tiny = load_preset(&dir.join("tinyimagenet_2024.yaml")).unwrap();

    assert_eq!(
        margin_row_fields_set(&cifar.margin_row),
        vec![
            "adaptive_reserve",
            "concurrent",
            "root_f32",
            "k_head",
            "k_trunk",
            "alpha_opt",
            "backward_interm",
            "bi_secs",
        ],
        "cifar100 margin_row field set changed; update the diff table above"
    );
    assert_eq!(
        margin_row_fields_set(&tiny.margin_row),
        vec![
            "reserve_secs",
            "concurrent",
            "root_f32",
            "tableau_f32",
            "backward_interm",
        ],
        "tinyimagenet margin_row field set changed; update the diff table above"
    );

    // Values, not just presence.
    assert_eq!(cifar.margin_row.adaptive_reserve, Some(true));
    assert_eq!(cifar.margin_row.k_head, Some(16));
    assert_eq!(cifar.margin_row.k_trunk, Some(16));
    assert_eq!(cifar.margin_row.bi_secs, Some(40));
    assert_eq!(tiny.margin_row.reserve_secs, Some(0));
    // #backward-interm budget is CIFAR-ONLY and the asymmetry is deliberate.
    // It was transplanted to tinyimagenet on 2026-08-22 and REVERTED the same
    // day, before it was ever scored (the yaml at tinyimagenet_2024.yaml:138
    // carries the reasoning and the pre-registered A/B). The commit that did
    // the revert ALSO left this loop asserting Some(40) for BOTH presets, so
    // the preset-diff safety net was red and asserting a value its own commit
    // had removed. Asserted asymmetrically now, which is the whole point of
    // this test.
    //
    // THE REVERT STANDS, AND IT HAS NOW BEEN MEASURED DIRECTLY — but both
    // justifications that stood here failed, the second by MISREADING THIS
    // PHASE'S OWN TELEMETRY:
    //
    // (1) The headroom argument (60 of 64 banked unsats publish at 89.5-97.1 s
    //     of 100 s) argued about the LANE, not this phase.
    // (2) The replacement said the phase "overruns its armed budget 2.11x
    //     median (195/195 logs, exhausted=true 195/195)" so "`bi_secs` does
    //     not control this phase". That read `[backward-interm] done: secs=`
    //     as phase cost; THAT CLOCK STARTS AT THE ROOT BUILD (`BiState::t0`),
    //     so it is build wall. Direct measurement 2026-08-24 (idx_5451,
    //     official 100 s): the phase's own compute is 2.83 s of a 48.9 s
    //     build, one 256-column pass is 0.64 s, and `exhausted=true` means
    //     the FORWARD BUILD spent the 20 s clock — the phase is STARVED at
    //     trunk layer 2 of ~10, not greedy. `work_secs=` on the done line now
    //     reports phase compute so the two cannot be confused again.
    //
    // The key then got its dose-response ON THIS CATEGORY
    // (docs/BI_DOSE_RESPONSE_5185_2026-08-24.md): the higher dose converted
    // banked-timeout idx_5185 only NONDETERMINISTICALLY (1/4; validation 0/3).
    // idx_9611 failed under the off, 20 s, and higher-dose arms alike on that
    // dirty binary, so it cannot be attributed to the dose. Tableau-f32 later
    // changed the root timing frame. Tiny stays absent pending a clean current-
    // binary 20-vs-30 A/B with the versioned execution receipt.
    assert_eq!(cifar.margin_row.bi_secs, Some(40), "cifar100 sets bi_secs");
    assert_eq!(
        tiny.margin_row.bi_secs, None,
        "tinyimagenet must NOT set bi_secs: the higher-dose 5185 result validated 0/3 \
         after one success, and current tableau-f32 timing needs a clean 20-vs-30 A/B \
         (docs/BI_DOSE_RESPONSE_5185_2026-08-24.md)"
    );
    let wall_presets = [
        ("cifar100_2024", &cifar.margin_row),
        ("tinyimagenet_2024", &tiny.margin_row),
    ];
    for (name, m) in wall_presets {
        assert_eq!(m.concurrent, Some(true), "{name} arms the concurrent lane");
        assert_eq!(m.root_f32, Some(true), "{name} arms the sound f32 root");
        assert_eq!(m.lru, None, "{name}: LRU stays on its measured default 64");
        assert_eq!(
            m.parallel, None,
            "{name}: parallel frontier stays on the measured all-worker default"
        );
        assert_eq!(
            m.backward_interm,
            Some(true),
            "{name} arms #backward-interm"
        );
        assert_eq!(
            m.k_adaptive, None,
            "{name}: typed delivery remains deliberately unarmed"
        );
        assert_eq!(
            m.epoch_tier0, None,
            "{name}: Tier-0 ranker remains default-off"
        );
        assert_eq!(
            m.epoch_universe, None,
            "{name}: Tier-0 universe stays on its declaration default"
        );
        assert_eq!(
            m.epoch_retain, None,
            "{name}: retention stays on its declaration default"
        );
        assert_eq!(m.epoch_depth, None, "{name}: epochs remain default-off");
        assert_eq!(
            m.epoch_attempts, None,
            "{name}: epoch attempts stay defaulted"
        );
        assert_eq!(
            m.epoch_parent_topk, None,
            "{name}: parent refresh remains dark"
        );
        assert_eq!(
            m.early_exit, None,
            "{name}: early exit rides its ON default"
        );
        assert_eq!(
            m.verifier_threads, None,
            "{name}: #verifier-thread-cap is unmeasured; ship it with an A/B"
        );
        assert_eq!(
            m.concurrent_threads, None,
            "{name}: concurrent lane width is unmeasured; omission preserves derived/global policy"
        );
    }

    // The diff is COMPLETE: no third shipped preset carries a margin_row key,
    // so nothing outside this table can inherit one of these decisions.
    let mut others: Vec<String> = Vec::new();
    for entry in fs::read_dir(&dir).expect("read configs/vnncomp25") {
        let path = entry.expect("dir entry").path();
        if path.extension().is_none_or(|e| e != "yaml") {
            continue;
        }
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .expect("preset stem")
            .to_string();
        if stem == "cifar100_2024" || stem == "tinyimagenet_2024" {
            continue;
        }
        let cfg = load_preset(&path).unwrap_or_else(|e| panic!("load {}: {e}", path.display()));
        let set = margin_row_fields_set(&cfg.margin_row);
        if !set.is_empty() {
            others.push(format!("{stem}: {set:?}"));
        }
    }
    assert!(
        others.is_empty(),
        "a third preset now sets margin_row keys, so the two-category diff above is \
         no longer the whole story: {others:?}"
    );
}

/// #tinyimagenet-alloc-parity-no-go: the CIFAR allocation trio regressed the
/// official-budget TinyImageNet sample from 5/10 to 2/10, including three lost
/// SAT rows. Pin the measured TinyImageNet baseline so a sibling-preset transfer
/// cannot silently override category-specific evidence again.
#[test]
fn tinyimagenet_2024_rejects_the_regressive_cifar100_allocation_trio() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let tiny = load_preset(&repo_root.join("configs/vnncomp25/tinyimagenet_2024.yaml")).unwrap();

    assert_eq!(tiny.bab.root_alpha_cap_secs, None);
    assert_eq!(tiny.bab.phase_budget.disjunctive_pgd_fraction, Some(0.40));
    assert_eq!(tiny.margin_row.adaptive_reserve, None);
    assert_eq!(
        tiny.margin_row.release_frac, None,
        "no margin-row reserve override should ride along with the rejected transfer"
    );

    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &tiny).unwrap();
    assert_eq!(config.root_alpha_cap_secs, None);
    assert_eq!(config.phase_budget.disjunctive_pgd_fraction, 0.40);

    // Bound-quality knobs remain independently scoped and are not part of this rollback.
    assert_eq!(tiny.bab.root_crown_interm_dense_head, None);
    assert_eq!(tiny.bab.clip.interm_domain, None);
    assert_eq!(tiny.bab.branching.kfsb_multi, None);
}

#[test]
fn cersyve_preset_enables_complete_clipping() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let preset = load_preset(&repo_root.join("configs/vnncomp25/cersyve.yaml")).unwrap();

    assert_eq!(
        preset.bab.clip.clip_type.as_deref(),
        Some("complete"),
        "cersyve preset should enable complete clipping"
    );

    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).unwrap();
    assert_eq!(
        config.input_clip_type,
        InputClipType::Complete,
        "complete clip_type should propagate to BetaCrownConfig"
    );
}

#[test]
fn lsnc_relu_stays_relaxed_while_cersyve_is_complete() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let lsnc_relu = load_preset(&repo_root.join("configs/vnncomp25/lsnc_relu.yaml")).unwrap();
    let cersyve = load_preset(&repo_root.join("configs/vnncomp25/cersyve.yaml")).unwrap();

    assert_eq!(
        lsnc_relu.bab.clip.clip_type.as_deref(),
        None,
        "lsnc_relu should rely on the default relaxed clip type"
    );
    assert_eq!(
        cersyve.bab.clip.clip_type.as_deref(),
        Some("complete"),
        "cersyve should keep its explicit complete clip type"
    );

    let mut lsnc_config = BetaCrownConfig::default();
    apply_preset(&mut lsnc_config, &lsnc_relu).unwrap();
    assert_eq!(
        lsnc_config.input_clip_type,
        InputClipType::Relaxed,
        "lsnc_relu should stay on the relaxed runtime clip path"
    );

    let mut cersyve_config = BetaCrownConfig::default();
    apply_preset(&mut cersyve_config, &cersyve).unwrap();
    assert_eq!(
        cersyve_config.input_clip_type,
        InputClipType::Complete,
        "cersyve should stay on the complete runtime clip path"
    );
}

#[test]
fn nn4sys_preset_matches_reference_input_split_schedule_4354() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let preset = load_preset(&repo_root.join("configs/vnncomp25/nn4sys.yaml")).unwrap();

    // pgd_order: skip — PGD disabled for nn4sys (Part of #4354)
    assert_eq!(preset.attack.pgd_order.as_deref(), Some("skip"));
    assert_eq!(
        preset.bab.phase_budget.post_bab_pgd_fraction,
        Some(0.0),
        "disabled NN4SYS PGD must not retain an engine tail reservation"
    );
    assert_eq!(
        preset.bab.phase_budget.vnncomp_post_bab_attack,
        Some(false),
        "the independent VNN-COMP post-BaB wrapper must follow the explicit proof-only policy"
    );
    assert_eq!(preset.bab.clip.relaxed, Some(true));
    assert_eq!(preset.bab.branching.method.as_deref(), Some("input"));
    assert_eq!(preset.bab.branching.input_split.enable, Some(true));
    assert_eq!(preset.bab.branching.input_split.sb_coeff_thresh, Some(0.1));
    assert_eq!(preset.bab.branching.input_split.reorder_bab, Some(true));

    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).unwrap();
    assert!(matches!(
        config.branching_heuristic,
        BranchingHeuristic::InputSplit
    ));
    assert!(
        config.enable_relaxed_clip,
        "nn4sys preset should propagate relaxed clipping into BetaCrownConfig"
    );
    assert_eq!(config.input_clip_type, InputClipType::Relaxed);
    assert_eq!(config.input_split_coeff_thresh, 0.1);
    assert!(config.reorder_bab);
    assert_eq!(
        config.phase_budget.post_bab_pgd_fraction, 0.0,
        "the applied NN4SYS engine schedule must give its full slice to proof"
    );
}

/// #nn4sys-seb-dark (B8): the shipped nn4sys preset arms Saturation-Escape
/// Branching through the TYPED key, so the dual-pool input brancher (and the
/// disjunctive precheck budget cap that funds it) is reachable without any
/// ambient `NY_SAT_ESCAPE_BRANCH` export — while a preset that never names the
/// key keeps the dark default byte-identically.
#[test]
fn nn4sys_preset_threads_sat_escape_branch_and_absence_stays_dark() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");

    // Key parses from the shipped nn4sys preset. It is `false`: `2e4af94b`
    // DISARMED it on a scored-wall-time A/B over the first 12 `_dual` rows —
    // armed proved 1/12 unsat, unarmed 8/12, so arming LOSES 7 rows. Arming
    // caps the per-clause precheck slice at `.min(0.5)` where the unarmed path
    // takes `.max(0.95)` (verify/disjunctive.rs:549-558), and on the mscn dual
    // band that precheck lane is what closes the disjuncts.
    //
    // This test previously asserted `Some(true)` and went stale the moment the
    // preset was corrected — it was pinning an intent the measurement refuted.
    // What is worth pinning is the WIRING in both directions, which is what it
    // now does.
    let nn4sys = load_preset(&repo_root.join("configs/vnncomp25/nn4sys.yaml")).unwrap();
    assert_eq!(
        nn4sys.bab.branching.input_split.sat_escape_branch,
        Some(false),
        "nn4sys deliberately DISARMS SEB (2e4af94b); NY_SAT_ESCAPE_BRANCH=1 still \
         forces it for experiments"
    );

    // …and an explicit `false` threads through to the engine field as disarmed.
    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &nn4sys).unwrap();
    assert!(
        !config.sat_escape_branch,
        "an explicit `false` must leave BetaCrownConfig::sat_escape_branch disarmed"
    );

    // Absent key = today's behavior: another input-split preset without the
    // key leaves the field at its dark default.
    let acasxu = load_preset(&repo_root.join("configs/vnncomp25/acasxu_2023.yaml")).unwrap();
    assert_eq!(acasxu.bab.branching.input_split.sat_escape_branch, None);
    let mut acasxu_config = BetaCrownConfig::default();
    apply_preset(&mut acasxu_config, &acasxu).unwrap();
    assert!(
        !acasxu_config.sat_escape_branch,
        "a preset that does not name the key must stay byte-identical (dark default)"
    );
}

#[test]
fn cersyve_and_lsnc_relu_presets_propagate_reference_input_split_tuning() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");

    let cersyve = load_preset(&repo_root.join("configs/vnncomp25/cersyve.yaml")).unwrap();
    assert_eq!(cersyve.bab.branching.input_split.enable, Some(true));
    assert_eq!(
        cersyve.bab.branching.input_split.sb_coeff_thresh,
        Some(1.0e-2)
    );
    assert_eq!(cersyve.bab.branching.input_split.sb_sum, Some(true));
    assert_eq!(
        cersyve.bab.branching.input_split.touch_zero_score,
        Some(0.1)
    );
    assert_eq!(
        cersyve.bab.branching.input_split.override_parallel,
        Some(true)
    );

    let lsnc_relu = load_preset(&repo_root.join("configs/vnncomp25/lsnc_relu.yaml")).unwrap();
    assert_eq!(lsnc_relu.bab.branching.input_split.enable, Some(true));
    assert_eq!(
        lsnc_relu.bab.branching.input_split.sb_coeff_thresh,
        Some(1.0e-2)
    );
    assert_eq!(lsnc_relu.bab.branching.input_split.sb_sum, Some(true));
    assert_eq!(
        lsnc_relu.bab.branching.input_split.touch_zero_score,
        Some(0.1)
    );
    assert_eq!(lsnc_relu.bab.branching.input_split.override_parallel, None);

    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &cersyve).unwrap();
    assert!(config.input_split_sb_sum);
    assert_eq!(config.input_split_coeff_thresh, 1.0e-2);
    assert_eq!(config.input_split_touch_zero_score, 0.1);
    assert!(config.input_split_override_parallel);
}

#[test]
fn cersyve_and_lsnc_relu_presets_enable_pgd_alpha_scale() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");

    for preset_name in ["cersyve", "lsnc_relu"] {
        let preset =
            load_preset(&repo_root.join(format!("configs/vnncomp25/{preset_name}.yaml"))).unwrap();
        let mut config = BetaCrownConfig::default();
        apply_preset(&mut config, &preset).unwrap();
        assert_eq!(
            config.pgd_optimizer,
            PgdOptimizer::SignedGradient,
            "{preset_name} should switch PGD to signed-gradient when pgd_alpha_scale=true"
        );
        assert_eq!(
            config.pgd_alpha_mode,
            PgdAlphaMode::InputRangeScaled(0.01),
            "{preset_name} should propagate pgd_alpha_scale into the runtime PGD alpha mode"
        );
    }
}

#[test]
fn cora_2024_preset_enables_clip_in_alpha_crown() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let preset = load_preset(&repo_root.join("configs/vnncomp25/cora_2024.yaml")).unwrap();

    assert_eq!(preset.bab.clip.in_alpha_crown, Some(true));

    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).unwrap();
    assert!(
        config.clip_in_alpha_crown,
        "cora_2024 preset should propagate clip_in_alpha_crown into BetaCrownConfig"
    );
}

#[test]
fn cora_and_metaroom_presets_explicitly_disable_interm_transfer_4358() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let cora = load_preset(&repo_root.join("configs/vnncomp25/cora_2024.yaml")).unwrap();
    let metaroom = load_preset(&repo_root.join("configs/vnncomp25/metaroom_2023.yaml")).unwrap();

    assert_eq!(cora.bab.interm_transfer, Some(false));
    assert_eq!(metaroom.bab.interm_transfer, Some(false));

    let mut cora_config = BetaCrownConfig::default();
    apply_preset(&mut cora_config, &cora).unwrap();
    assert!(
        !cora_config.enable_interm_transfer,
        "cora_2024 preset should override the enabled default and keep interm_transfer disabled"
    );

    let mut metaroom_config = BetaCrownConfig::default();
    apply_preset(&mut metaroom_config, &metaroom).unwrap();
    assert!(
        !metaroom_config.enable_interm_transfer,
        "metaroom_2023 preset should keep its explicit interm_transfer=false override"
    );
}

#[test]
fn metaroom_preset_propagates_dd_zonotope_admission_overrides() {
    // #metaroom-ddzono: the scored entry point sets no env vars, so the
    // category preset is the only way the dd-zonotope admission caps can
    // reach `DdZonoConfig` for metaroom's 5376-input / k=161 instances.
    //
    // MECHANISM test: an inline section must propagate every knob.
    let inline: PresetConfig = serde_yaml::from_str(
        r#"
dd_zonotope:
  min_input_numel: 5000
  max_k: 192
  max_generators: 4096
  interm_intersect: true
"#,
    )
    .expect("dd_zonotope preset section must parse");
    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &inline).unwrap();
    assert_eq!(config.dd_zonotope_min_input_numel, Some(5000));
    assert_eq!(config.dd_zonotope_max_k, Some(192));
    assert_eq!(config.dd_zonotope_max_generators, Some(4096));
    assert_eq!(config.dd_zonotope_collect_interm, Some(true));

    // SHIPPED-STATE test: the metaroom yaml keeps the section DISARMED until
    // the near-wall A/B (spec_idx_28-class rows vs the ~18s/row cost) and the
    // campaign-GPU interm-intersect value test are both recorded — the
    // population-evidence discipline. Re-point these at Some(..) ONLY together
    // with those measurements.
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let metaroom = load_preset(&repo_root.join("configs/vnncomp25/metaroom_2023.yaml")).unwrap();
    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &metaroom).unwrap();
    assert_eq!(config.dd_zonotope_min_input_numel, None);
    assert_eq!(config.dd_zonotope_max_k, None);
    assert_eq!(config.dd_zonotope_max_generators, None);
    assert_eq!(config.dd_zonotope_collect_interm, None);

    // Every other shipped preset leaves the section absent, which must remain
    // byte-identical to the built-in caps: apply changes nothing.
    let acasxu = load_preset(&repo_root.join("configs/vnncomp25/acasxu_2023.yaml")).unwrap();
    let mut untouched = BetaCrownConfig::default();
    apply_preset(&mut untouched, &acasxu).unwrap();
    assert_eq!(untouched.dd_zonotope_min_input_numel, None);
    assert_eq!(untouched.dd_zonotope_max_k, None);
    assert_eq!(untouched.dd_zonotope_max_generators, None);
    assert_eq!(untouched.dd_zonotope_collect_interm, None);
}

#[test]
fn metaroom_preset_arms_the_cz_hybrid_instance_allowlist() {
    // #cz-hybrid: the scored entry point sets no env vars, so the typed
    // preset section is the ONLY route by which the wired CZ+Box verdict lane
    // (`commands::cz_metaroom_wired`) can engage. The boundary's own unit
    // tests cover the admission mechanism; this pins the SHIPPED state:
    // metaroom_2023 arms exactly the qualified spec_idx_119 instance, and
    // every other preset leaves the section absent = lane dark.
    //
    // Changing either name here without re-qualifying the sealed replays
    // would make the lane engage on an instance it must fail closed on — the
    // catch_unwind boundary makes that a wasted-budget bug rather than a
    // soundness bug, but do not do it.
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let metaroom = load_preset(&repo_root.join("configs/vnncomp25/metaroom_2023.yaml")).unwrap();
    assert_eq!(
        metaroom.cz_hybrid.only_onnx.as_deref(),
        Some("6cnn_ry_39_6_no_custom_OP.onnx")
    );
    assert_eq!(
        metaroom.cz_hybrid.only_vnnlib.as_deref(),
        Some("spec_idx_119_eps_0.00000436.vnnlib")
    );

    // Every other shipped preset must leave the lane dark.
    let acasxu = load_preset(&repo_root.join("configs/vnncomp25/acasxu_2023.yaml")).unwrap();
    assert_eq!(acasxu.cz_hybrid.only_onnx, None);
    assert_eq!(acasxu.cz_hybrid.only_vnnlib, None);
}

#[test]
fn cersyve_and_lsnc_relu_presets_propagate_min_batch_fill_ratio() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");

    let cersyve = load_preset(&repo_root.join("configs/vnncomp25/cersyve.yaml")).unwrap();
    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &cersyve).unwrap();
    assert_eq!(
        config.min_batch_fill_ratio, 0.1,
        "cersyve min_batch_fill_ratio should match reference (0.1)"
    );

    let lsnc_relu = load_preset(&repo_root.join("configs/vnncomp25/lsnc_relu.yaml")).unwrap();
    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &lsnc_relu).unwrap();
    assert_eq!(
        config.min_batch_fill_ratio, 0.0,
        "lsnc_relu min_batch_fill_ratio should match reference (0.0)"
    );
}

#[test]
fn acasxu_2023_preset_matches_hardcoded_config() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let preset = load_preset(&repo_root.join("configs/vnncomp25/acasxu_2023.yaml")).unwrap();

    // Verify YAML fields parsed correctly
    assert_eq!(
        preset.solver.bound_prop_method.as_deref(),
        Some("crown"),
        "acasxu preset should use plain CROWN (not alpha-CROWN)"
    );
    assert_eq!(preset.solver.batch_size, Some(16384));
    assert_eq!(preset.attack.pgd_order.as_deref(), Some("after"));
    assert_eq!(preset.attack.pgd_restarts, Some(10000));
    assert_eq!(preset.bab.branching.method, Some("input".to_string()));
    assert_eq!(preset.bab.branching.input_split.reorder_bab, Some(true));
    assert_eq!(preset.bab.clip.relaxed, Some(true));

    // Apply and verify config matches BetaCrownConfig::acas_xu()
    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).unwrap();

    let reference = BetaCrownConfig::acas_xu();
    assert_eq!(
        config.use_alpha_crown, reference.use_alpha_crown,
        "preset should disable alpha-CROWN like acas_xu() config"
    );
    assert_eq!(
        config.batch_size, reference.batch_size,
        "preset batch_size should match acas_xu() config"
    );
    assert_eq!(
        config.branching_heuristic, reference.branching_heuristic,
        "preset branching should match acas_xu() config"
    );
    assert_eq!(
        config.reorder_bab, reference.reorder_bab,
        "preset input-split reordering should match acas_xu() config"
    );
    assert_eq!(
        config.input_split_depth, reference.input_split_depth,
        "preset multi-dim input_split_depth should match acas_xu() config"
    );
    assert_eq!(
        config.enable_relaxed_clip, reference.enable_relaxed_clip,
        "preset relaxed_clip should match acas_xu() config"
    );
    assert_eq!(
        config.enable_pgd_attack, reference.enable_pgd_attack,
        "preset pgd_attack should match acas_xu() config"
    );
    assert_eq!(
        config.pgd_restarts, reference.pgd_restarts,
        "preset pgd_restarts should match acas_xu() config"
    );
}

/// vit_2023 regression pin: softmax "complex" stays DISABLED in the shipped
/// preset — measured net-negative at vit scale (00338ba) — while the Phase A
/// alpha/share_alphas knobs stay in place and apply_preset succeeds cleanly.
#[test]
fn vit_2023_preset_keeps_softmax_complex_disabled() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let preset = load_preset(&repo_root.join("configs/vnncomp25/vit_2023.yaml")).unwrap();

    // softmax: complex is deliberately NOT set — measured net-negative at vit
    // scale (00338ba: ~3x per-iteration cost, best bound flat while the
    // direct-LSE arm improves). The machinery stays in-tree; the preset
    // re-enables it only after the recorded follow-ups land.
    assert_eq!(
        preset.solver.alpha_crown.softmax.as_deref(),
        None,
        "vit_2023 preset keeps softmax-complex disabled (measured net-negative; see 00338ba)"
    );
    assert_eq!(preset.solver.alpha_crown.iterations, Some(50));
    assert_eq!(preset.solver.alpha_crown.lr_alpha, Some(0.5));
    assert_eq!(preset.solver.alpha_crown.share_alphas, Some(true));

    // The softmax field is consumed at model load (graph rewrite), not by
    // BetaCrownConfig; applying the preset must still succeed cleanly.
    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).unwrap();
    assert_eq!(config.alpha_config.iterations, 50);
    assert_eq!(config.alpha_config.learning_rate, 0.5);

    // #disj-demand (measured 2026-08-26, official 100 s): both vit model
    // families are outside the exact-gradient fragment, so the 5% disjunctive
    // attack slice can only run SPSA over point-IBP forwards and has never
    // produced a counterexample on this family. The typed key must arm the
    // demand-zeroing through the yaml route (no NY_* involved), while the
    // configured fraction is retained as the disarmed fallback.
    assert_eq!(
        preset.bab.phase_budget.demand_sized_disjunctive_pgd,
        Some(true)
    );
    assert!(config.phase_budget.demand_sized_disjunctive_pgd);
    assert!((config.phase_budget.disjunctive_pgd_fraction - 0.05).abs() < 1e-6);
}

/// cgan_2023 root-collection scheduling caps (#4413, #cgan-bn11-budget):
/// the preset's per-node CROWN-IBP cap and aggregate α reference-refresh cap
/// flow through to the exact typed configs consumed by the collectors. The
/// per-node floor and refresh fraction retain their built-in defaults.
#[test]
fn cgan_2023_preset_raises_crown_ibp_per_node_cap() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let preset = load_preset(&repo_root.join("configs/vnncomp25/cgan_2023.yaml")).unwrap();
    let load_config =
        build_onnx_load_config(&preset).expect("official cGAN loader policy must be admitted");
    assert_eq!(
        load_config.batch_norm_folding_policy(),
        ny_onnx::BatchNormFoldingPolicy::PreserveRaw,
        "official cGAN verdicts must target the authored graph"
    );
    assert!(
        load_config.require_authored_float32_initializers(),
        "official cGAN loading must fail before verification on rewritten FLOAT initializers"
    );
    assert_eq!(
        preset.model.forward_linear_spec_alpha,
        Some(false),
        "the candidate remains typed and dark pending the physical moat"
    );

    assert_eq!(preset.bab.crown_ibp_per_node_cap_secs, Some(150.0));
    assert_eq!(preset.bab.crown_ibp_per_node_floor_secs, None);
    // #cgan-abc-parity (2026-08-27): reorder_bab is production - both an abc
    // treatment field and the prerequisite of the exact-domain input clip.
    assert_eq!(preset.bab.branching.input_split.reorder_bab, Some(true));
    assert_eq!(preset.bab.branching.input_split.warm_parallel, None);
    assert_eq!(preset.bab.branching.input_split.alpha_iteration, Some(1));
    assert_eq!(preset.bab.branching.input_split.lr_alpha, Some(0.05));
    assert_eq!(preset.bab.alpha_crown.reference_refresh_max_secs, Some(12));
    assert_eq!(preset.bab.alpha_crown.reference_refresh_fraction, None);
    assert_eq!(
        preset
            .bab
            .alpha_crown
            .forward_linear_deadline_fallback_to_ibp,
        Some(true)
    );

    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).unwrap();
    assert_eq!(config.crown_ibp_per_node_cap_secs, Some(150.0));
    assert_eq!(config.crown_ibp_per_node_floor_secs, None);
    assert!(config.reorder_bab);
    assert!(!config.input_split_warm_parallel);
    assert_eq!(config.input_split_alpha_iteration, 1);
    assert_eq!(config.input_split_lr_alpha, 0.05);
    assert_eq!(config.alpha_config.reference_refresh_max_secs, Some(12));
    assert!(config.alpha_config.forward_linear_deadline_fallback_to_ibp);
    assert_eq!(
        config.alpha_config.reference_refresh_fraction,
        ny_propagate::AlphaCrownConfig::DEFAULT_REFERENCE_REFRESH_FRACTION
    );

    let budget = config.crown_ibp_per_node_time_budget();
    assert_eq!(budget.cap_secs, Some(150.0));
    assert_eq!(budget.floor_secs, None);

    // Companion phase-budget levers: retained SAT instances fall to PGD in
    // under 2 s, while the official holdout otherwise spent 85.1 s before BaB
    // and reserved another 10% for an unsuccessful post-BaB attack.
    assert_eq!(config.phase_budget.disjunctive_pgd_fraction, 0.10);
    assert_eq!(config.phase_budget.disjunctive_pgd_max_secs, Some(5));
    assert_eq!(config.phase_budget.post_bab_pgd_fraction, 0.0);
    assert_eq!(config.phase_budget.initial_bounds_fraction, 0.45);
    // The current forward-linear precheck returns as soon as its shared map is
    // available, so this remains a ceiling rather than spent time.
    assert_eq!(config.phase_budget.disjunctive_precheck_fraction, 0.85);
}

/// The forward-alpha treatment is a typed, reviewable preset rather than an
/// environment overlay. Normalize its single treatment field and require the
/// full serialized presets to match, so sealed A/B runs cannot accidentally
/// drift a loader, attack, or solver knob.
#[test]
fn cgan_forward_alpha_surrogate_canary_differs_by_exactly_one_typed_field() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let production = load_preset(&repo_root.join("configs/vnncomp25/cgan_2023.yaml")).unwrap();
    let canary = load_preset(
        &repo_root.join("configs/vnncomp25/cgan_2023_forward_alpha_surrogate_canary.yaml"),
    )
    .unwrap();

    assert_eq!(production.model.forward_linear_spec_alpha, Some(false));
    assert_eq!(canary.model.forward_linear_spec_alpha, Some(true));

    for (name, preset) in [("production", &production), ("canary", &canary)] {
        let load_config = build_onnx_load_config(preset)
            .unwrap_or_else(|error| panic!("{name} cGAN loader policy must be admitted: {error}"));
        assert_eq!(
            load_config.batch_norm_folding_policy(),
            ny_onnx::BatchNormFoldingPolicy::PreserveRaw,
            "{name} cGAN preset must verify the authored graph"
        );
        assert!(
            load_config.require_authored_float32_initializers(),
            "{name} cGAN preset must reject rewritten authored FLOAT initializers"
        );
    }

    let mut normalized_canary = canary;
    normalized_canary.model.forward_linear_spec_alpha = production.model.forward_linear_spec_alpha;
    assert_eq!(
        serde_json::to_value(&normalized_canary).unwrap(),
        serde_json::to_value(&production).unwrap(),
        "forward-alpha canary may differ from shipped cGAN only at forward_linear_spec_alpha"
    );
}

#[test]
fn cgan_2026_preset_preserves_authored_graph_and_keeps_surrogate_dark() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let preset = load_preset(&repo_root.join("configs/vnncomp26/cgan2026.yaml")).unwrap();
    let load_config = build_onnx_load_config(&preset).expect("official 2026 cGAN loader policy");
    assert_eq!(
        load_config.batch_norm_folding_policy(),
        ny_onnx::BatchNormFoldingPolicy::PreserveRaw
    );
    assert!(load_config.require_authored_float32_initializers());
    assert_eq!(preset.model.forward_linear_spec_alpha, Some(false));
}

/// The warm-parallel route is intentionally absent from the shipped cGAN
/// preset. Its canary changes exactly the reordered-loop and scoped activation
/// fields, so a row-7 serial/parallel A/B cannot drift any other solver knob.
/// The target-complete root collector remains a sidecar experiment. Keep its
/// treatment surface attributable: it may receive more initial-bound time,
/// request exact root intermediates, and arm the new typed policy, but must
/// preserve every other production cGAN setting.
#[test]
fn cgan_sparse_target_complete_canary_differs_by_exactly_three_knobs() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let production = load_preset(&repo_root.join("configs/vnncomp25/cgan_2023.yaml")).unwrap();
    let canary = load_preset(
        &repo_root.join("configs/vnncomp25/cgan_2023_sparse_target_complete_canary.yaml"),
    )
    .unwrap();

    assert_eq!(
        production.bab.phase_budget.initial_bounds_fraction,
        Some(0.45)
    );
    assert_eq!(canary.bab.phase_budget.initial_bounds_fraction, Some(0.85));
    assert_eq!(production.bab.alpha_crown.fix_interm_bounds, None);
    assert_eq!(canary.bab.alpha_crown.fix_interm_bounds, Some(false));
    assert_eq!(
        production.bab.alpha_crown.cgan_sparse_target_complete_root,
        None
    );
    assert_eq!(
        canary.bab.alpha_crown.cgan_sparse_target_complete_root,
        Some(true)
    );
    assert_eq!(
        canary
            .bab
            .alpha_crown
            .forward_linear_deadline_fallback_to_ibp,
        Some(true),
        "the canary must preserve production's certified deadline fallback"
    );

    let mut normalized_canary = canary;
    normalized_canary.bab.phase_budget.initial_bounds_fraction =
        production.bab.phase_budget.initial_bounds_fraction;
    normalized_canary.bab.alpha_crown.fix_interm_bounds =
        production.bab.alpha_crown.fix_interm_bounds;
    normalized_canary
        .bab
        .alpha_crown
        .cgan_sparse_target_complete_root =
        production.bab.alpha_crown.cgan_sparse_target_complete_root;
    assert_eq!(
        serde_json::to_value(&normalized_canary).unwrap(),
        serde_json::to_value(&production).unwrap(),
        "canary may differ only at initial_bounds_fraction, fix_interm_bounds, and \
         cgan_sparse_target_complete_root"
    );
}

/// cgan2026 resamples properties over the same seven ONNX models as cgan_2023.
/// Keep model-cost and sound-fallback scheduling levers aligned without
/// transferring any result-count claim between competitions.
#[test]
fn cgan2026_mirrors_byte_identical_model_schedule_levers() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let cgan_2025 = load_preset(&repo_root.join("configs/vnncomp25/cgan_2023.yaml")).unwrap();
    let cgan_2026 = load_preset(&repo_root.join("configs/vnncomp26/cgan2026.yaml")).unwrap();

    assert_ne!(cgan_2026.general.root_path, cgan_2025.general.root_path);
    assert!(
        cgan_2026
            .general
            .root_path
            .as_deref()
            .is_some_and(|path| path.ends_with("cgan2026/1.0")),
        "the regular-track cgan2026 preset must name the selected VNN-LIB 1.0 manifest"
    );
    assert_eq!(cgan_2026.general.device.as_deref(), Some("wgpu"));
    assert_eq!(
        cgan_2026.bab.crown_ibp_per_node_cap_secs,
        cgan_2025.bab.crown_ibp_per_node_cap_secs
    );
    assert_eq!(
        serde_json::to_value(&cgan_2026.bab.phase_budget).unwrap(),
        serde_json::to_value(&cgan_2025.bab.phase_budget).unwrap(),
        "byte-identical model schedules must retain the proven phase ceilings"
    );
    assert_eq!(
        serde_json::to_value(&cgan_2026.bab.branching).unwrap(),
        serde_json::to_value(&cgan_2025.bab.branching).unwrap(),
        "byte-identical model schedules must retain the certified input-split options"
    );
    assert_eq!(
        cgan_2026.bab.alpha_crown.reference_refresh_max_secs,
        Some(12)
    );
    assert_eq!(cgan_2026.bab.alpha_crown.reference_refresh_fraction, None);
    assert_eq!(
        cgan_2026
            .bab
            .alpha_crown
            .forward_linear_deadline_fallback_to_ibp,
        Some(true)
    );
    assert_eq!(
        cgan_2026.bab.alpha_crown.cgan_complete_crown_ibp_root,
        cgan_2025.bab.alpha_crown.cgan_complete_crown_ibp_root,
        "byte-identical cGAN models must retain the certified complete-root policy"
    );

    let mut normalized_2026 = cgan_2026.clone();
    normalized_2026.general.root_path = cgan_2025.general.root_path.clone();
    normalized_2026.general.device = cgan_2025.general.device.clone();
    assert_eq!(
        serde_json::to_value(&normalized_2026).unwrap(),
        serde_json::to_value(&cgan_2025).unwrap(),
        "byte-identical cGAN presets may differ only in corpus path and selected device"
    );

    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &cgan_2026).unwrap();
    assert_eq!(config.crown_ibp_per_node_cap_secs, Some(150.0));
    assert_eq!(config.phase_budget.disjunctive_pgd_fraction, 0.10);
    assert_eq!(config.phase_budget.disjunctive_pgd_max_secs, Some(5));
    assert_eq!(config.phase_budget.attack_extension_fraction, 0.0);
    assert_eq!(config.phase_budget.initial_bounds_fraction, 0.45);
    assert_eq!(config.phase_budget.disjunctive_precheck_fraction, 0.85);
    assert_eq!(config.phase_budget.post_bab_pgd_fraction, 0.0);
    assert_eq!(config.branching_heuristic, BranchingHeuristic::InputSplit);
    assert_eq!(config.fsb_candidates, 5);
    assert!(config.reorder_bab);
    assert_eq!(config.input_split_alpha_iteration, 1);
    assert_eq!(config.input_split_lr_alpha, 0.05);
    assert!(config.input_split_ibp_enhancement);
    assert!(config.input_split_stacked_rebound);
    assert_eq!(config.alpha_config.reference_refresh_max_secs, Some(12));
    assert!(config.alpha_config.forward_linear_deadline_fallback_to_ibp);
    assert!(config.alpha_config.cgan_complete_crown_ibp_root);
}

/// The warm-parallel route is intentionally absent from the shipped cGAN
/// preset. Its canary changes exactly the measured reordered-loop batch and
/// scoped activation fields, so a row-7 A/B cannot drift any other solver
/// knob. Since #cgan-abc-parity (2026-08-27) promoted `reorder_bab: true`
/// into production, the canary's remaining treatment surface is batch_size
/// and warm_parallel.
#[test]
fn cgan_warm_parallel_batch8_canary_is_scoped_and_differs_by_exactly_two_knobs() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let production = load_preset(&repo_root.join("configs/vnncomp25/cgan_2023.yaml")).unwrap();
    let canary = load_preset(
        &repo_root.join("configs/vnncomp25/cgan_2023_reorder_warm_parallel_canary.yaml"),
    )
    .unwrap();

    assert_eq!(production.bab.branching.input_split.reorder_bab, Some(true));
    assert_eq!(production.bab.branching.input_split.warm_parallel, None);
    assert_eq!(production.bab.batch_size, Some(64));
    assert_eq!(canary.bab.batch_size, Some(8));
    assert_eq!(canary.bab.branching.input_split.reorder_bab, Some(true));
    assert_eq!(canary.bab.branching.input_split.warm_parallel, Some(true));

    let mut normalized_canary = canary.clone();
    normalized_canary.bab.batch_size = production.bab.batch_size;
    normalized_canary.bab.branching.input_split.warm_parallel =
        production.bab.branching.input_split.warm_parallel;
    assert_eq!(
        serde_json::to_value(&normalized_canary).unwrap(),
        serde_json::to_value(&production).unwrap(),
        "canary may differ only at batch_size and warm_parallel"
    );

    let mut production_config = BetaCrownConfig::default();
    apply_preset(&mut production_config, &production).unwrap();
    let mut canary_config = BetaCrownConfig::default();
    apply_preset(&mut canary_config, &canary).unwrap();
    assert!(production_config.reorder_bab);
    assert!(!production_config.input_split_warm_parallel);
    assert_eq!(production_config.batch_size, 64);
    assert!(canary_config.reorder_bab);
    assert!(canary_config.input_split_warm_parallel);
    assert_eq!(canary_config.batch_size, 8);
}

/// This default-off canary began as a seven-knob translation of
/// alpha-beta-CROWN's current cGAN recipe. #cgan-abc-parity (2026-08-27)
/// promoted five of those knobs into production itself (conv_mode,
/// bound_prop_method, sb_coeff_thresh, reorder_bab, adv_check) after the
/// promoted set converted `nCh_1_prop_3` (banked timeout,895 -> unsat in
/// 513-533 s, 3 reproductions) with the 12-sat + test_nano moat clean. The
/// canary's remaining treatment surface is device and pgd_restarts. Keep it
/// exact: any extra drift would make a sealed row-7 A/B impossible to
/// attribute.
#[test]
fn cgan_abcrown_parity_canary_differs_by_exactly_two_knobs() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let production = load_preset(&repo_root.join("configs/vnncomp25/cgan_2023.yaml")).unwrap();
    let canary =
        load_preset(&repo_root.join("configs/vnncomp25/cgan_2023_abcrown_parity_canary.yaml"))
            .unwrap();

    // The two remaining treatments.
    assert_eq!(production.general.device, None);
    assert_eq!(canary.general.device.as_deref(), Some("wgpu"));
    assert_eq!(production.attack.pgd_restarts, Some(200));
    assert_eq!(canary.attack.pgd_restarts, Some(100));

    // The five promoted knobs: now production values, shared by the canary.
    assert_eq!(
        production.general.conv_mode,
        Some(ny_propagate::ConvMode::Matrix)
    );
    assert_eq!(
        production.solver.bound_prop_method.as_deref(),
        Some("crown")
    );
    assert_eq!(
        production.bab.branching.input_split.sb_coeff_thresh,
        Some(1.0e-2)
    );
    assert_eq!(production.bab.branching.input_split.reorder_bab, Some(true));
    // Sentinel semantics are intentionally counterintuitive: zero enables the
    // periodic SAT-only probe immediately, while -1 disables it.
    assert_eq!(production.bab.branching.input_split.adv_check, Some(0));
    assert_eq!(canary.general.conv_mode, production.general.conv_mode);
    assert_eq!(
        canary.solver.bound_prop_method,
        production.solver.bound_prop_method
    );
    assert_eq!(
        canary.bab.branching.input_split.sb_coeff_thresh,
        production.bab.branching.input_split.sb_coeff_thresh
    );
    assert_eq!(
        canary.bab.branching.input_split.reorder_bab,
        production.bab.branching.input_split.reorder_bab
    );
    assert_eq!(
        canary.bab.branching.input_split.adv_check,
        production.bab.branching.input_split.adv_check
    );

    let mut normalized_canary = canary.clone();
    normalized_canary.general.device = production.general.device.clone();
    normalized_canary.attack.pgd_restarts = production.attack.pgd_restarts;
    assert_eq!(
        serde_json::to_value(&normalized_canary).unwrap(),
        serde_json::to_value(&production).unwrap(),
        "alpha-beta-CROWN parity canary may differ from shipped cGAN only at device and pgd_restarts"
    );

    let mut production_config = BetaCrownConfig::default();
    apply_preset(&mut production_config, &production).unwrap();
    let mut canary_config = BetaCrownConfig::default();
    apply_preset(&mut canary_config, &canary).unwrap();

    assert_eq!(production_config.conv_mode, ny_propagate::ConvMode::Matrix);
    assert!(!production_config.use_alpha_crown);
    assert_eq!(production_config.pgd_restarts, 200);
    assert_eq!(production_config.input_split_coeff_thresh, 1.0e-2);
    assert!(production_config.reorder_bab);
    assert_eq!(production_config.adv_check, 0);
    assert!(production_config.input_split_fresh_domain_clip);

    assert_eq!(canary_config.conv_mode, ny_propagate::ConvMode::Matrix);
    assert!(!canary_config.use_alpha_crown);
    assert_eq!(canary_config.pgd_restarts, 100);
    assert_eq!(canary_config.input_split_coeff_thresh, 1.0e-2);
    assert!(canary_config.reorder_bab);
    assert_eq!(canary_config.adv_check, 0);
    assert!(canary_config.input_split_fresh_domain_clip);
}

/// The default-off generator-prefix treatment must be a pure routing delta
/// from the alpha-beta-CROWN parity canary: explicit ReLU splitting replaces
/// explicit input splitting, and no input-split-only tuning leaks into the
/// treatment. Its sidecar filename also keeps normal cgan_2023 preset
/// discovery from selecting it.
#[test]
fn cgan_generator_relu_phase_split_canary_is_explicit_and_scoped() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let parity_path = repo_root.join("configs/vnncomp25/cgan_2023_abcrown_parity_canary.yaml");
    let canary_path =
        repo_root.join("configs/vnncomp25/cgan_2023_generator_relu_phase_split_canary.yaml");

    assert_ne!(
        canary_path.file_name().and_then(|name| name.to_str()),
        Some("cgan_2023.yaml"),
        "default category discovery must not silently select the canary"
    );

    let parity = load_preset(&parity_path).unwrap();
    let canary = load_preset(&canary_path).unwrap();
    assert_eq!(parity.bab.branching.method.as_deref(), Some("input"));
    assert_eq!(canary.bab.branching.method.as_deref(), Some("relu"));
    assert_eq!(
        serde_json::to_value(&canary.bab.branching.input_split).unwrap(),
        serde_json::to_value(InputSplitPreset::default()).unwrap(),
        "ReLU treatment must not retain inapplicable input-split tuning"
    );

    let resolved = resolve_branching(&canary)
        .unwrap()
        .expect("explicit relu method must resolve");
    assert_eq!(resolved.heuristic, BranchingHeuristic::LargestBoundWidth);
    assert!(
        resolved.use_relu_split,
        "explicit relu method must route to ReLU splitting, not auto-selection"
    );

    // The exact-domain input clip is InputSplit-only (`BetaCrownConfig::validate`
    // rejects it under any other branching heuristic), so the ReLU-splitting
    // canary must NOT carry the parity/production `input_split_fresh_domain_clip`
    // key: it is inapplicable input-split tuning of exactly the kind the
    // stanza-removal rule above exists for.
    assert_eq!(canary.bab.clip.input_split_fresh_domain_clip, None);

    let mut normalized_canary = canary.clone();
    normalized_canary.bab.branching.method = parity.bab.branching.method.clone();
    normalized_canary.bab.branching.input_split = parity.bab.branching.input_split.clone();
    normalized_canary.bab.clip.input_split_fresh_domain_clip =
        parity.bab.clip.input_split_fresh_domain_clip;
    assert_eq!(
        serde_json::to_value(&normalized_canary).unwrap(),
        serde_json::to_value(&parity).unwrap(),
        "generator-prefix canary may differ from the parity canary only at the explicit branching route and removed input-split-only tuning (stanza + exact-domain clip)"
    );

    assert_eq!(canary.general.device.as_deref(), Some("wgpu"));
    assert_eq!(
        canary.general.conv_mode,
        Some(ny_propagate::ConvMode::Matrix)
    );
    assert_eq!(canary.solver.bound_prop_method.as_deref(), Some("crown"));
    assert_eq!(canary.attack.pgd_restarts, Some(100));
    assert_eq!(canary.bab.batch_size, Some(64));
    assert_eq!(canary.bab.phase_budget.disjunctive_pgd_fraction, Some(0.10));
    assert_eq!(canary.bab.phase_budget.attack_extension_fraction, Some(0.0));
    assert_eq!(canary.bab.phase_budget.initial_bounds_fraction, Some(0.45));
    assert_eq!(
        canary.bab.phase_budget.disjunctive_precheck_fraction,
        Some(0.85)
    );
    assert_eq!(canary.bab.clip.relaxed, Some(true));
    assert_eq!(canary.bab.clip.relaxed_iterations, Some(20));
    assert_eq!(canary.bab.alpha_crown.lr_alpha, Some(0.2));
    assert_eq!(canary.bab.alpha_crown.iterations, Some(100));
    assert_eq!(canary.bab.beta_crown.lr_alpha, Some(0.1));
    assert_eq!(canary.bab.beta_crown.lr_beta, Some(0.15));
    assert_eq!(canary.bab.beta_crown.iterations, Some(25));
}

/// The alpha-beta-CROWN VGG recipe is a default-off canary. Keep NY's shipped
/// WGPU/root phase-budget tuning and isolate exactly the five treatment knobs.
#[test]
fn vgg_abcrown_parity_canary_is_default_off_and_scoped() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let production = load_preset(&repo_root.join("configs/vnncomp25/vggnet16_2022.yaml")).unwrap();
    let canary =
        load_preset(&repo_root.join("configs/vnncomp25/vggnet16_2022_abcrown_parity_canary.yaml"))
            .unwrap();

    assert_eq!(production.model.vgg_abcrown_treatment, None);
    assert_eq!(canary.model.vgg_abcrown_treatment, Some(true));
    assert_eq!(
        canary.solver.bound_prop_method.as_deref(),
        Some("forward+backward")
    );
    assert_eq!(canary.bab.branching.method.as_deref(), Some("sb"));
    assert_eq!(canary.bab.branching.input_split.enable, Some(true));
    assert_eq!(canary.attack.pgd_order.as_deref(), Some("input_bab"));

    let mut normalized_canary = canary.clone();
    normalized_canary.model.vgg_abcrown_treatment = production.model.vgg_abcrown_treatment;
    normalized_canary.solver.bound_prop_method = production.solver.bound_prop_method.clone();
    normalized_canary.bab.branching.method = production.bab.branching.method.clone();
    normalized_canary.bab.branching.input_split.enable =
        production.bab.branching.input_split.enable;
    normalized_canary.attack.pgd_order = production.attack.pgd_order.clone();
    assert_eq!(
        serde_json::to_value(&normalized_canary).unwrap(),
        serde_json::to_value(&production).unwrap(),
        "VGG canary may differ from production only at its five treatment fields"
    );

    let mut production_config = BetaCrownConfig::default();
    apply_preset(&mut production_config, &production).unwrap();
    let mut canary_config = BetaCrownConfig::default();
    apply_preset(&mut canary_config, &canary).unwrap();

    assert!(
        production_config.use_alpha_crown,
        "shipped VGG root optimizations remain unchanged"
    );
    assert!(!production_config.use_forward_bounds);
    assert_eq!(
        production_config.branching_heuristic,
        BranchingHeuristic::LargestBoundWidth
    );

    assert!(!canary_config.use_alpha_crown);
    assert!(canary_config.use_forward_bounds);
    assert_eq!(
        canary_config.branching_heuristic,
        BranchingHeuristic::InputSplit
    );
    assert!(canary_config.enable_pgd_attack);
    assert_eq!(canary_config.batch_size, production_config.batch_size);
    assert_eq!(
        serde_json::to_value(&canary_config.phase_budget).unwrap(),
        serde_json::to_value(&production_config.phase_budget).unwrap(),
        "existing root phase budgets must survive the treatment"
    );
}

/// The production YOLO preset deliberately omits a branching method. Once its
/// loaded-model dimensions reach auto-resolution, that must select kFSB and
/// carry the typed critical-row policy into the final verifier config. The
/// companion ny-propagate selector test proves this config bit reaches the real
/// scorer with `NY_MO_SCORER_FIX` absent.
#[test]
fn yolo_2023_preset_auto_kfsb_arms_typed_critical_row_policy() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let preset = load_preset(&repo_root.join("configs/vnncomp25/yolo_2023.yaml")).unwrap();
    assert_eq!(
        preset.bab.branching.method, None,
        "YOLO must reach model-aware auto resolution"
    );

    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).unwrap();
    assert!(
        !config.use_multi_objective_critical_kfsb,
        "the reusable preset alone must not globally arm the costly scorer"
    );

    let resolved = resolve_auto_branching(
        AutoBranchingRequest {
            mip_complete_verifier: false,
            spec_input_count: Some(8112),
        },
        ModelStructure {
            param_count: 5_000_000,
            has_conv: true,
            relu_node_count: 40,
            is_dag: true,
        },
        8112,
    );
    assert_eq!(resolved.heuristic, BranchingHeuristic::Kfsb);
    assert_eq!(resolved.reason, AutoBranchingReason::HighDimOrManyRelu);
    assert!(resolved.use_multi_objective_critical_kfsb);

    // Mirrors the two runtime stamps in handle_beta_crown_command after preset
    // application and auto resolution.
    config.branching_heuristic = resolved.heuristic.clone();
    apply_resolved_auto_branching_runtime_policy(&mut config, &resolved);
    assert_eq!(config.branching_heuristic, BranchingHeuristic::Kfsb);
    assert!(config.use_multi_objective_critical_kfsb);
}

/// yolo_2023 pins the historical 12-second base cap after an official 2026 row
/// showed that every useful target completed within 8.715 seconds while two
/// later targets consumed 51.772 and 31.142 seconds before falling back to IBP.
#[test]
fn yolo_2023_preset_caps_doomed_crown_ibp_targets() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let preset = load_preset(&repo_root.join("configs/vnncomp25/yolo_2023.yaml")).unwrap();

    assert_eq!(preset.bab.crown_ibp_per_node_cap_secs, Some(12.0));

    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).unwrap();
    assert_eq!(config.crown_ibp_per_node_cap_secs, Some(12.0));
    assert_eq!(config.crown_ibp_per_node_time_budget().cap_secs, Some(12.0));
}

/// Presets that do not opt in carry no per-node overrides: the collector keeps
/// the 2.0-second floor and adaptive remaining-budget cap (e.g. cersyve,
/// dist_shift).
#[test]
fn presets_without_crown_ibp_budget_knob_keep_defaults() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    for name in ["cersyve.yaml", "dist_shift_2023.yaml"] {
        let preset = load_preset(&repo_root.join("configs/vnncomp25").join(name)).unwrap();
        assert_eq!(
            preset.bab.crown_ibp_per_node_cap_secs, None,
            "{name} must not set the cgan-only cap knob"
        );
        assert_eq!(preset.bab.crown_ibp_per_node_floor_secs, None);

        let mut config = BetaCrownConfig::default();
        apply_preset(&mut config, &preset).unwrap();
        let budget = config.crown_ibp_per_node_time_budget();
        assert_eq!(budget, ny_propagate::CrownIbpPerNodeTimeBudget::default());
    }
}

/// TEETH for #reclaim-pgd on relusplitter.
///
/// The disjunctive PGD lane on this benchmark is RESTART-bound, not
/// fraction-bound: `pgd_restarts: 50` x the default `pgd_steps: 50` x 9 clauses
/// exhausts restarts before the fractional deadline is reached (measured
/// 2026-08-01: 11.5-27.5 s used of a 30 s/90 s slice on GT-unsat oval21 rows). So
/// `disjunctive_pgd_fraction` is the WRONG control here and an ABSOLUTE cap is the
/// right one.
///
/// Deleting the preset's cap fails the first assertion. "Fixing" it by lowering the
/// fraction instead fails the second. Reaching for any other phase knob — in
/// particular one that shortens a BOUND phase, which could change a verdict — fails
/// the whole-struct comparison, which is the assertion that actually pins the blast
/// radius.
#[test]
fn relusplitter_caps_the_restart_bound_disjunctive_pgd_phase() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let preset = load_preset(&repo_root.join("configs/vnncomp25/relusplitter.yaml")).unwrap();

    assert_eq!(
        preset.bab.phase_budget.disjunctive_pgd_max_secs,
        Some(5),
        "relusplitter must cap the restart-bound disjunctive PGD phase in ABSOLUTE \
         seconds; the reclaimed budget flows to BaB via the ledger's remaining()"
    );
    assert_eq!(
        preset.bab.phase_budget.disjunctive_pgd_fraction, None,
        "the fraction is measured INERT on relusplitter (the lane never reaches its \
         fractional deadline) — it must not be used as the control here"
    );

    // #proof-tail (2026-08-21 receipts in the preset): the outer wrapper lane
    // is explicitly disabled alongside the zero engine fraction so the router
    // arms PresetProofTail (the cersyve precedent) and the solver keeps
    // everything but the fixed publication margin.
    assert_eq!(
        preset.bab.phase_budget.vnncomp_post_bab_attack,
        Some(false),
        "relusplitter must disable the outer post-BaB wrapper lane explicitly; \
         paired with post_bab_pgd_fraction 0.0 this arms the PresetProofTail route"
    );

    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).unwrap();

    // SCOPE, asserted as a whole-struct diff: the preset moves EXACTLY two phase
    // budget fields. Every other phase — initial bounds, reduced verification,
    // MIP, precheck — keeps its sealed default, so no bound-producing phase is
    // shortened and no verdict can move. Both moved fields bound ATTACK phases:
    // shortening an attack can only lose a counterexample (`sat` -> `unknown`,
    // a completeness cost); it can never mint an `unsat`, because no `unsat` is
    // ever concluded from a failed attack. The post-BaB tail is measured dead
    // weight on this category (full 220 sweep, 2026-08-21: all 20 GT-sat rows
    // land upfront at <=0.3-0.9s; the tail converted zero rows while holding
    // ~10s of every 60s row and ~21s of every 180s row).
    let expected = ny_propagate::PhaseBudgetConfig {
        disjunctive_pgd_max_secs: Some(5),
        post_bab_pgd_fraction: 0.0,
        ..Default::default()
    };
    assert_eq!(
        config.phase_budget, expected,
        "relusplitter's phase budget must differ from the sealed defaults in exactly \
         disjunctive_pgd_max_secs and post_bab_pgd_fraction"
    );
}

/// Collins' absolute cap suppresses an optional forward-linear warmer whose
/// cold build dominates the small CNN proof. The cap is schedule-only: neither
/// its attack work nor its optional bound-cache warmer can grant uncertified
/// authority, and the whole-struct assertion pins every other proof/attack
/// phase to its sealed default.
#[test]
fn collins_rul_caps_the_disjunctive_prewave_without_fraction_drift() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let preset =
        load_preset(&repo_root.join("configs/vnncomp25/collins_rul_cnn_2022.yaml")).unwrap();

    assert_eq!(
        preset.bab.phase_budget.disjunctive_pgd_max_secs,
        Some(15),
        "Collins must keep the measured absolute prewave ceiling at every timeout tier"
    );
    assert_eq!(
        preset.bab.phase_budget.disjunctive_pgd_fraction, None,
        "a fractional override grows at long tiers and can re-admit the optional warmer"
    );

    let mut config = BetaCrownConfig::default();
    apply_preset(&mut config, &preset).unwrap();
    let expected = ny_propagate::PhaseBudgetConfig {
        disjunctive_pgd_max_secs: Some(15),
        ..Default::default()
    };
    assert_eq!(
        config.phase_budget, expected,
        "Collins may change only the attack prewave's absolute ceiling"
    );
}

/// The cap is SCOPED: presets that never measured the restart-bound lane must keep
/// the pure-fraction behaviour, so landing relusplitter's cap cannot silently
/// shorten falsification on an unrelated benchmark.
#[test]
fn disjunctive_pgd_cap_stays_scoped_to_presets_that_measured_it() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    for name in ["acasxu_2023.yaml", "cersyve.yaml", "vit_2023.yaml"] {
        let preset = load_preset(&repo_root.join("configs/vnncomp25").join(name)).unwrap();
        assert_eq!(
            preset.bab.phase_budget.disjunctive_pgd_max_secs, None,
            "{name} must keep the pure-fraction disjunctive PGD behaviour"
        );

        let mut config = BetaCrownConfig::default();
        apply_preset(&mut config, &preset).unwrap();
        assert_eq!(
            config.phase_budget.disjunctive_pgd_max_secs, None,
            "{name} must not inherit relusplitter's absolute cap"
        );
    }
}
