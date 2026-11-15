// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! The §1.7 enforcement ratchet (ENGINE_PROGRAM, #soundness-spine M2b).
//!
//! The proof profile may choose artifact sinks and audit rates, never
//! certification. That is only true if it is MECHANICAL, so each rule is a
//! source check:
//!
//! 1. no certifier module names the proof profile (or its audit rates);
//! 2. every `ProofSink` method returns `()` (sinks are write-only);
//! 3. `UnsatToken` has exactly one constructor, with exactly two callers,
//!    and no module outside `certify` mints a legacy-uncovered token; every
//!    adapter that mints a cover payload is `pub(crate)`, and the one public
//!    exact-emptiness entry re-checks what it is given;
//! 4. `cublasSetMathMode` is called only by the handle's math-mode pin, and
//!    only with `CUBLAS_DEFAULT_MATH` (the verdict-feeding IBP widening relies
//!    on IEEE round-to-nearest on the shared handle);
//! 5. `certify` never names the TF32 search GEMM, nor fused multiply-add.
//!
//! Needles are assembled at runtime, so this file cannot match itself, and
//! every scanned file must be non-trivially long: an empty module compiles
//! and scans clean, which would turn a check into a no-op.

use std::path::{Path, PathBuf};

fn crate_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn read(path: &Path) -> String {
    let text =
        std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    assert!(
        text.len() > 200,
        "{} is suspiciously short ({} bytes): an empty module compiles and scans clean",
        path.display(),
        text.len()
    );
    text
}

/// `text` without comment lines: documentation (including `compile_fail`
/// examples that spell forbidden calls) is not code.
fn code_only(text: &str) -> String {
    text.lines()
        .filter(|line| !line.trim_start().starts_with("//"))
        .collect::<Vec<_>>()
        .join("\n")
}

fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = std::fs::read_dir(dir).unwrap_or_else(|e| panic!("read {}: {e}", dir.display()));
    for entry in entries {
        let path = entry.expect("dir entry").path();
        if path.is_dir() {
            rust_files(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
    out.sort();
}

/// The certify facade's own sources (its test module excluded).
fn certify_sources() -> Vec<PathBuf> {
    let mut files = Vec::new();
    rust_files(&crate_root().join("src/certify"), &mut files);
    files.retain(|p| p.file_name().is_some_and(|n| n != "tests.rs"));
    assert!(
        files.len() >= 6,
        "expected the certify facade's modules, found {files:?}"
    );
    files
}

#[test]
fn proof_profile_ratchet_certifiers_never_name_the_profile() {
    let needles = [
        format!("{}{}", "Proof", "Profile"),
        format!("{}_{}", "proof", "profile"),
        format!("{}{}", "Audit", "Rates"),
    ];
    let mut files = certify_sources();
    for rel in [
        "src/margin_row/zono_state/problem.rs",
        "src/margin_row/engine.rs",
        "src/margin_row/errlane.rs",
        "src/layers/linear/crown_single.rs",
        // #root-box-source (ENGINE_PROGRAM M5b): the patch-backward root box
        // source and the two certified builders it composes.
        "src/margin_row/root_box.rs",
        "src/margin_row/patch.rs",
        "src/margin_row/ibp_seed.rs",
        // #wave-beta-bab (ENGINE_PROGRAM M6): the wave tree's frozen -> clip
        // -> y-pack leaf ladder, the program's new TCB composition item 5.
        "src/margin_row/bab/wave.rs",
        // #frozen-dual-bab + #fd-block-eval + #wide-fd-search: the tier that
        // certifies prunes and runs G4. Its cross-check RATE arrives from the
        // lane as a plain `u32` (`TierPolicy::crosscheck_every`, set in
        // `bab.rs` from the entry point's audit rates), so the tier itself
        // must never learn which profile chose it.
        "src/margin_row/zono_bab.rs",
        "src/margin_row/zono_bab/wide.rs",
    ] {
        files.push(crate_root().join(rel));
    }
    let problem_dir = crate_root().join("src/margin_row/zono_state/problem");
    if problem_dir.is_dir() {
        rust_files(&problem_dir, &mut files);
    }
    // #router-r1 (ROUTER_PROGRAM §7): the runtime router's policy computes the
    // same plan in both profiles, so it may never learn which one runs either
    // (its own ratchet, ny-core `tests/router_ratchet.rs`, checks code lines;
    // this one checks whole files).
    let router_dir = crate_root().join("../ny-core/src/router");
    assert!(
        router_dir.is_dir(),
        "{} is missing: the router scan would be VOID",
        router_dir.display()
    );
    rust_files(&router_dir, &mut files);
    for file in &files {
        let text = read(file);
        for needle in &needles {
            assert!(
                !text.contains(needle.as_str()),
                "certifier module {} names `{needle}`: the profile may select sinks and rates \
                 only, never certification",
                file.display()
            );
        }
    }
}

/// G4 (ENGINE_PROGRAM M2d, §1.7 B4): the frozen-dual tier's block-path
/// cross-check rate is read from the ENTRY POINT's audit rates by the lane
/// (`bab.rs`) and reaches the tier only as the policy's plain `u32`. So Proof
/// re-checks every prune, Competition one in 256, and the tier (scanned by the
/// certifier test above) never learns which profile chose the rate. Needles
/// are assembled at runtime; the declaration default may appear only at its
/// three declaration sites (`SearchOptions::default`, `TierPolicy::clamped`,
/// the test-only rate probe).
#[test]
fn proof_profile_ratchet_g4_rate_comes_from_the_audit_rates() {
    let bab = code_only(&read(&crate_root().join("src/margin_row/bab.rs")));
    let read_rates = format!("{}::{}()", "proof_profile", "audit_rates");
    let rate = format!("{}.{}", "rates", "crosscheck_every");
    assert!(
        bab.contains(read_rates.as_str()) && bab.contains(rate.as_str()),
        "the lane no longer reads G4's rate from the entry point's audit rates"
    );
    let handed = format!("{}()", "fd_crosscheck_every");
    assert!(
        bab.matches(handed.as_str()).count() >= 3,
        "the lane must define the rate read once and hand it to BOTH frozen-dual tier sites"
    );
    let tier = code_only(&read(&crate_root().join("src/margin_row/zono_bab.rs")));
    let from_policy = format!(
        "{}: {}.{}",
        "crosscheck_every", "policy", "crosscheck_every"
    );
    assert!(
        tier.contains(from_policy.as_str()),
        "the tier's class search must take the policy's G4 rate"
    );
    let hard = format!(
        "{}: {},",
        "crosscheck_every", "FROZEN_DUAL_CROSSCHECK_EVERY"
    );
    assert_eq!(
        tier.matches(hard.as_str()).count(),
        3,
        "the declaration default may be the rate only at its three declaration sites; a \
         fourth hard-codes G4 past the proof profile"
    );
}

#[test]
fn proof_profile_ratchet_proof_sink_methods_return_unit() {
    let text = code_only(&read(&crate_root().join("src/certify/sink.rs")));
    let decl = format!("pub trait {}", "ProofSink");
    let start = text.find(decl.as_str()).expect("the ProofSink trait");
    let open = start + text[start..].find('{').expect("trait body");
    let mut depth = 0usize;
    let mut close = None;
    for (i, c) in text[open..].char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    close = Some(open + i);
                    break;
                }
            }
            _ => {}
        }
    }
    let body = &text[open..close.expect("closed trait body")];
    let mut methods = 0usize;
    let mut rest = body;
    while let Some(at) = rest.find("fn ") {
        let sig_end = rest[at..].find(';').map_or(rest.len(), |e| at + e);
        let sig = &rest[at..sig_end];
        assert!(
            !sig.contains("->"),
            "ProofSink method returns a value (sinks must be write-only): {sig}"
        );
        methods += 1;
        rest = &rest[sig_end..];
    }
    assert!(methods >= 2, "found only {methods} ProofSink methods");
}

#[test]
fn proof_profile_ratchet_unsat_token_has_one_constructor() {
    let token = code_only(&read(&crate_root().join("src/certify/token.rs")));
    let def = format!("fn {}(", "mint");
    let call = format!("{}::{}(", "UnsatToken", "mint");
    assert_eq!(token.matches(def.as_str()).count(), 1, "one constructor");
    assert_eq!(
        token.matches(call.as_str()).count(),
        2,
        "minted only by authorize_unsat and legacy_uncovered_unsat"
    );
    for file in certify_sources() {
        if file.ends_with("token.rs") {
            continue;
        }
        assert!(
            !code_only(&read(&file)).contains(call.as_str()),
            "{} mints a token outside token.rs",
            file.display()
        );
    }
    let legacy = format!("{}(", "legacy_uncovered_unsat");
    let mut files = Vec::new();
    rust_files(&crate_root().join("src"), &mut files);
    for file in files {
        if file.components().any(|c| c.as_os_str() == "certify") {
            continue;
        }
        let text = std::fs::read_to_string(&file).expect("read source");
        assert!(
            !code_only(&text).contains(legacy.as_str()),
            "{} mints a legacy-uncovered unsat token outside the certify facade",
            file.display()
        );
    }
}

/// The margin-row adapters that MINT (`Certified`, `LeafEvidence`,
/// `ClassCover`, `MarginRowValue`) trust their caller to pass a value the
/// certified evaluator produced, so they must stay `pub(crate)`: with them
/// `pub`, any crate (the `ny-cli` publication path included) could build a
/// publishable `UnsatToken` from a float of its choosing. Checked by name
/// (the named adapters must exist AS `pub(crate)`, so a rename fails loudly)
/// and by shape (no `pub fn` in the adapter module may return a minting type).
#[test]
fn proof_profile_ratchet_minting_adapters_are_crate_private() {
    let text = code_only(&read(&crate_root().join("src/certify/margin_row.rs")));
    for name in [
        "classify",
        "outward_bound",
        "outward_emptiness",
        "bound_evidence",
        "empty_evidence",
        "root_class",
        "frozen_dual_class",
    ] {
        let private = format!("{}(crate) fn {name}(", "pub");
        let public = format!("{} fn {name}(", "pub");
        assert_eq!(
            text.matches(private.as_str()).count(),
            1,
            "the minting adapter `{name}` must exist exactly once as `pub(crate)`"
        );
        assert!(
            !text.contains(public.as_str()),
            "the minting adapter `{name}` is `pub`: another crate could mint a certified value"
        );
    }
    let minting = [
        format!("{}<", "Certified"),
        "LeafEvidence".to_string(),
        "ClassCover".to_string(),
        "MarginRowValue".to_string(),
    ];
    let public_fn = format!("{} fn ", "pub");
    let mut rest = text.as_str();
    let mut public_fns = 0usize;
    while let Some(at) = rest.find(public_fn.as_str()) {
        let sig_end = rest[at..].find('{').map_or(rest.len(), |e| at + e);
        let sig = &rest[at..sig_end];
        for needle in &minting {
            assert!(
                !sig.contains(needle.as_str()),
                "a `pub fn` in certify::margin_row mints `{needle}` for any caller: {sig}"
            );
        }
        public_fns += 1;
        rest = &rest[sig_end..];
    }
    assert!(
        public_fns >= 2,
        "expected the public non-minting adapters (`closes`, `leaf_path`), found {public_fns}"
    );
}

/// #spine-legacy-lanes: the exact specification-emptiness module mints an
/// `ExactEmptiness` only through its `pub(crate)` checker, and its ONE public
/// entry re-checks every group through that checker before `authorize_unsat`
/// sees a cover. No public function of the module hands out an
/// `ExactEmptiness`, a `ClassCover` or a `Certified` value, so another crate can
/// obtain nothing but a token for atoms the facade itself proved empty.
#[test]
fn proof_profile_ratchet_exact_emptiness_mints_only_after_its_own_check() {
    let text = code_only(&read(&crate_root().join("src/certify/exact.rs")));
    let checker = format!("{}(crate) fn {}(", "pub", "scalar_emptiness");
    assert_eq!(
        text.matches(checker.as_str()).count(),
        1,
        "the exact-emptiness checker must exist exactly once as `pub(crate)`"
    );
    let public_checker = format!("{} fn {}(", "pub", "scalar_emptiness");
    assert!(
        !text.contains(public_checker.as_str()),
        "the exact-emptiness checker is `pub`: another crate could mint a certificate"
    );
    // Every `ExactEmptiness {` that is not the struct definition or its
    // `impl` block is a struct literal; exactly one may exist (the checker's).
    let literal = format!("{} {{", "ExactEmptiness");
    let declarations = [format!("struct {literal}"), format!("impl {literal}")];
    let constructions = text
        .match_indices(literal.as_str())
        .filter(|&(at, _)| {
            !declarations
                .iter()
                .any(|decl| text[..at + literal.len()].ends_with(decl.as_str()))
        })
        .count();
    assert_eq!(
        constructions, 1,
        "an `ExactEmptiness` literal outside the checker: {constructions} found"
    );
    let checker_at = text.find(checker.as_str()).expect("checker");
    let literal_at = text
        .match_indices(literal.as_str())
        .map(|(at, _)| at)
        .find(|&at| {
            !declarations
                .iter()
                .any(|decl| text[..at + literal.len()].ends_with(decl.as_str()))
        })
        .expect("one literal");
    let checker_end = text[checker_at..]
        .find("\n}\n")
        .map_or(text.len(), |e| checker_at + e);
    assert!(
        checker_at < literal_at && literal_at < checker_end,
        "the one `ExactEmptiness` literal must sit inside the checker"
    );
    let minting = [
        "ExactEmptiness".to_string(),
        "ClassCover".to_string(),
        format!("{}<", "Certified"),
    ];
    let public_fn = format!("{} fn ", "pub");
    let entry = format!("{} fn {}(", "pub", "authorize_scalar_emptiness");
    let mut rest = text.as_str();
    let mut public_fns = Vec::new();
    while let Some(at) = rest.find(public_fn.as_str()) {
        let sig_end = rest[at..].find('{').map_or(rest.len(), |e| at + e);
        let sig = &rest[at..sig_end];
        for needle in &minting {
            assert!(
                !sig.contains(needle.as_str()),
                "a `pub fn` in certify::exact hands out `{needle}`: {sig}"
            );
        }
        public_fns.push(sig.to_string());
        rest = &rest[sig_end..];
    }
    assert!(
        public_fns.iter().any(|sig| sig.starts_with(entry.as_str())),
        "the public exact-emptiness entry is missing: {public_fns:?}"
    );
    let body_start = text.find(entry.as_str()).expect("entry");
    let body = &text[body_start..];
    let body = &body[..body.find("\n}\n").map_or(body.len(), |e| e + 3)];
    let recheck = format!("{}(atoms)", "scalar_emptiness");
    assert!(
        body.contains(recheck.as_str()),
        "the public entry must re-check every group through the checker"
    );
}

/// #graph-bab-cover-ledger: the graph-lane adapters that MINT closing
/// evidence (`graph_bound_evidence`, `graph_infeasible_evidence`,
/// `graph_clip_evidence`, `graph_screen_evidence`, and #vit-v3-bankable's
/// `graph_lower_evidence`, `graph_emptiness_evidence`) trust their caller to pass
/// what the lane's own evaluator decided, so they are `pub(crate)`; no `pub fn`
/// of `certify::graph` hands out a `Certified`, a `LeafEvidence`, a
/// `ClassCover`, a `TreeCover` or a `GraphInvocationCover`; the cover gates'
/// `TreeCover` constructors are not `pub`; and the module's one public token
/// door re-checks entailment and box containment and mints through
/// `authorize_unsat` (so `UnsatToken::mint` keeps its two callers).
#[test]
fn proof_profile_ratchet_graph_minting_adapters_are_crate_private() {
    let text = code_only(&read(&crate_root().join("src/certify/graph.rs")));
    for name in [
        "graph_bound_evidence",
        "graph_infeasible_evidence",
        "graph_clip_evidence",
        "graph_screen_evidence",
        // #vit-v3-bankable: the multi-objective frame's lower-only root
        // evidence and the graph emptiness slack.
        "graph_lower_evidence",
        "graph_emptiness_evidence",
    ] {
        let private = format!("{}(crate) fn {name}(", "pub");
        let public = format!("{} fn {name}(", "pub");
        assert_eq!(
            text.matches(private.as_str()).count(),
            1,
            "the graph minting adapter `{name}` must exist exactly once as `pub(crate)`"
        );
        assert!(
            !text.contains(public.as_str()),
            "the graph minting adapter `{name}` is `pub`: another crate could mint evidence"
        );
    }
    let minting = [
        format!("{}<", "Certified"),
        "LeafEvidence".to_string(),
        "ClassCover".to_string(),
        "TreeCover".to_string(),
        format!("{}{}", "GraphInvocation", "Cover"),
    ];
    let public_fn = format!("{} fn ", "pub");
    let door = format!("{} fn {}(", "pub", "authorize_graph_instance");
    let mut rest = text.as_str();
    let mut public_fns = Vec::new();
    while let Some(at) = rest.find(public_fn.as_str()) {
        let sig_end = rest[at..].find('{').map_or(rest.len(), |e| at + e);
        let sig = &rest[at..sig_end];
        for needle in &minting {
            assert!(
                !sig.contains(needle.as_str()),
                "a `pub fn` in certify::graph hands out `{needle}`: {sig}"
            );
        }
        public_fns.push(sig.to_string());
        rest = &rest[sig_end..];
    }
    assert!(
        public_fns.iter().any(|sig| sig.starts_with(door.as_str())),
        "the graph instance door is missing: {public_fns:?}"
    );
    let start = text.find(door.as_str()).expect("door");
    let body = &text[start..];
    let body = &body[..body.find("\n}\n").map_or(body.len(), |e| e + 3)];
    for needle in [
        format!("{}(", "authorize_unsat"),
        format!(".{}(", "entails"),
        format!(".{}(", "contains_box"),
    ] {
        assert!(
            body.contains(needle.as_str()),
            "the graph instance door must re-check `{needle}` itself"
        );
    }
    let cover = code_only(&read(&crate_root().join("src/certify/cover.rs")));
    for name in ["checked", "relabeled"] {
        let public = format!("{} fn {name}(", "pub");
        assert!(
            !cover.contains(public.as_str()),
            "`TreeCover::{name}` is `pub`: another crate could assemble a cover"
        );
    }
}

#[test]
fn proof_profile_ratchet_math_mode_is_pinned_once() {
    let call = format!("sys::{}(", "cublasSetMathMode");
    let pin = format!("fn {}(", "pin_default_math_mode");
    let default_math = format!("{}_{}", "CUBLAS_DEFAULT", "MATH");
    let mut files = Vec::new();
    rust_files(&crate_root().join("../ny-cuda/src"), &mut files);
    assert!(!files.is_empty(), "ny-cuda sources not found");
    let mut sites = Vec::new();
    for file in &files {
        let text = code_only(&std::fs::read_to_string(file).expect("read source"));
        for (at, _) in text.match_indices(call.as_str()) {
            sites.push((file.clone(), text.clone(), at));
        }
    }
    assert_eq!(
        sites.len(),
        1,
        "cublasSetMathMode may be called only by the math-mode pin: {:?}",
        sites.iter().map(|(f, _, _)| f).collect::<Vec<_>>()
    );
    let (file, text, at) = &sites[0];
    let fn_at = text
        .find(pin.as_str())
        .unwrap_or_else(|| panic!("{} has no {pin}", file.display()));
    let next_fn = text[fn_at + pin.len()..]
        .find("\nfn ")
        .map_or(text.len(), |e| fn_at + pin.len() + e);
    assert!(
        fn_at < *at && *at < next_fn,
        "the one cublasSetMathMode call must sit inside {pin}"
    );
    let line_end = text[*at..].find(';').map_or(text.len(), |e| at + e);
    assert!(
        text[*at..line_end].contains(default_math.as_str()),
        "the pin must set {default_math} and nothing else"
    );
}

#[test]
fn proof_profile_ratchet_certify_never_names_tf32_gemm_or_fma() {
    let needles = [
        format!("{}_{}", "gemm_f32", "fast"),
        format!("{}{}", "mul_", "add"),
        format!(".{}(", "fma"),
    ];
    for file in certify_sources() {
        let text = read(&file);
        for needle in &needles {
            assert!(
                !text.contains(needle.as_str()),
                "{} contains `{needle}`",
                file.display()
            );
        }
    }
}

/// #root-box-source (ENGINE_PROGRAM M5b), R7: the certified patch-backward
/// root box source and the certified builders it composes (the IBP reference,
/// the patch walk) are portable scalar Rust — no fused multiply-add, no TF32
/// search GEMM — so their Higham charges and the column-local bit-identity
/// they rely on hold on every host.
#[test]
fn proof_profile_ratchet_root_box_source_never_names_tf32_gemm_or_fma() {
    let needles = [
        format!("{}_{}", "gemm_f32", "fast"),
        format!("{}{}", "mul_", "add"),
        format!(".{}(", "fma"),
    ];
    for rel in [
        "src/margin_row/root_box.rs",
        "src/margin_row/patch.rs",
        "src/margin_row/ibp_seed.rs",
    ] {
        let file = crate_root().join(rel);
        let text = read(&file);
        for needle in &needles {
            assert!(
                !text.contains(needle.as_str()),
                "{} contains `{needle}`",
                file.display()
            );
        }
    }
}

/// #wave-beta-bab (ENGINE_PROGRAM M6), R7: the wave tree's certify ladder
/// (frozen -> clip -> y-pack, new TCB composition item 5 of §1.6) is portable
/// scalar Rust that only calls the lane's existing certified evaluators — no
/// fused multiply-add and no TF32 search GEMM — and the untrusted search it
/// consumes reaches it only through `ny_core::search` (never `ny_bbe`).
#[test]
fn proof_profile_ratchet_wave_ladder_never_names_tf32_gemm_or_fma() {
    let needles = [
        format!("{}_{}", "gemm_f32", "fast"),
        format!("{}{}", "mul_", "add"),
        format!(".{}(", "fma"),
        format!("{}_{}", "ny", "bbe"),
    ];
    let file = crate_root().join("src/margin_row/bab/wave.rs");
    let text = read(&file);
    assert!(
        text.contains("fn wave_certify_on("),
        "{} no longer holds the certify ladder — the scan is VOID",
        file.display()
    );
    let code = code_only(&text);
    for needle in &needles {
        assert!(
            !code.contains(needle.as_str()),
            "{} contains `{needle}`",
            file.display()
        );
    }
}

/// ENGINE_PROGRAM §1.7 B1 (proof bundle v2): the frame's hint ledger is
/// WRITE-ONLY. Its fields are private, and every public method of
/// `impl HintLedger` returns nothing, `Self` (its constructor) or `bool` (the
/// capture flag) — no accessor can hand hint data back to a verdict path.
/// Positive control: the scanner sees a returning method when one exists.
#[test]
fn hint_ledger_exposes_no_reader() {
    let text = code_only(&read(&crate_root().join("src/certify/sink.rs")));
    let decl = format!("pub struct {} {{", "HintLedger");
    let at = text.find(decl.as_str()).expect("the HintLedger struct");
    let body_end = at + text[at..].find("\n}").expect("struct body end");
    let body = &text[at + decl.len()..body_end];
    assert!(
        !body.contains("pub "),
        "HintLedger fields must be private: {body}"
    );
    let imp = format!("impl {} {{", "HintLedger");
    let start = text.find(imp.as_str()).expect("impl HintLedger");
    let mut depth = 0usize;
    let mut end = None;
    for (i, c) in text[start..].char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    end = Some(start + i);
                    break;
                }
            }
            _ => {}
        }
    }
    let block = &text[start..end.expect("impl end")];
    let check = |block: &str| -> Vec<String> {
        let mut bad = Vec::new();
        let mut methods = 0usize;
        let mut rest = block;
        let needle = format!("{} fn ", "pub");
        while let Some(at) = rest.find(needle.as_str()) {
            let sig_end = rest[at..].find('{').map_or(rest.len(), |e| at + e);
            let sig = &rest[at..sig_end];
            methods += 1;
            if let Some(ret) = sig.split("->").nth(1) {
                let ret = ret.trim();
                if ret != "Self" && ret != "bool" {
                    bad.push(sig.to_string());
                }
            }
            rest = &rest[sig_end..];
        }
        if methods < 5 {
            bad.push(format!(
                "only {methods} public methods found: the scan is VOID"
            ));
        }
        bad
    };
    let bad = check(block);
    assert!(bad.is_empty(), "HintLedger exposes a reader: {bad:?}");
    let probe = format!(
        "impl X {{ {p} fn new() -> Self {{}} {p} fn a(&self) -> bool {{}} {p} fn b(&mut self) {{}} \
         {p} fn c(&mut self) {{}} {p} fn leaks(&self) -> &[u8] {{}} }}",
        p = "pub"
    );
    assert_eq!(
        check(&probe).len(),
        1,
        "positive control: the scan sees a reader"
    );
}

/// B2.2 (ENGINE_PROGRAM §1.7 B1): a lane's proof TAPE is write-only like the
/// hint ledger. Every method of `impl TapeRecorder` returns nothing or its
/// constructor's `Option<Self>`, and every public method of
/// `impl LaneProofCapture` returns nothing, a `bool` or a count — no accessor
/// hands a recorded leaf back to a verdict path (the leaves leave only through
/// the write-only sink, after publication). Positive control: the scan sees a
/// returning method when one exists.
#[test]
fn lane_proof_tape_exposes_no_reader() {
    let text = code_only(&read(&crate_root().join("src/certify/lane_proof.rs")));
    let block_of = |imp: &str| -> String {
        let start = text
            .find(imp)
            .unwrap_or_else(|| panic!("`{imp}` not found"));
        let mut depth = 0usize;
        for (i, c) in text[start..].char_indices() {
            match c {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        return text[start..start + i].to_string();
                    }
                }
                _ => {}
            }
        }
        panic!("`{imp}` has no end");
    };
    let check = |block: &str, allowed: &[&str], min: usize| -> Vec<String> {
        let mut bad = Vec::new();
        let mut methods = 0usize;
        let mut rest = block;
        while let Some(at) = rest.find("fn ") {
            let sig_end = rest[at..].find('{').map_or(rest.len(), |e| at + e);
            let sig = &rest[at..sig_end];
            methods += 1;
            if let Some(ret) = sig.split("->").nth(1) {
                let ret = ret.trim();
                if !allowed.contains(&ret) {
                    bad.push(sig.to_string());
                }
            }
            rest = &rest[sig_end..];
        }
        if methods < min {
            bad.push(format!("only {methods} methods found: the scan is VOID"));
        }
        bad
    };
    let recorder = block_of(&format!("impl {} {{", "TapeRecorder"));
    let bad = check(
        &recorder,
        &[
            "Option<Self>",
            "(Vec<TapeLit>, Option<&'static str>)",
            "Vec<f64>",
            "Self",
            "usize",
        ],
        8,
    );
    assert!(bad.is_empty(), "TapeRecorder exposes a reader: {bad:?}");
    // The two private helpers that return data build a RECORD from the
    // lane's own history (never read a recorded leaf): they must stay private.
    for helper in ["fn lits_of(", "fn beta_of("] {
        let at = recorder.find(helper).expect("helper present");
        let head = &recorder[at.saturating_sub(12)..at];
        assert!(
            !head.contains("pub"),
            "`{helper}` must stay private to the recorder"
        );
    }
    let capture = block_of(&format!("impl {} {{", "LaneProofCapture"));
    let bad = check(&capture, &["usize", "bool"], 3);
    assert!(bad.is_empty(), "LaneProofCapture exposes a reader: {bad:?}");
    let probe = "impl X { fn a(&self) -> usize {} fn b(&self) -> bool {} fn c(&mut self) {} \
                 fn leaks(&self) -> &[u8] {} }";
    assert_eq!(
        check(probe, &["usize", "bool"], 3).len(),
        1,
        "positive control: the scan sees a reader"
    );
}

/// B2.3 (ENGINE_PROGRAM §1.7 B1): the frozen-dual tier's prune TRACE is
/// write-only like the hint ledger. Its fields are private, its constructor
/// is crate-private, and every public method of `impl FdProofTrace` returns
/// a class index or a flag — no accessor can hand a recorded prune back to a
/// verdict path; the prunes leave only through `stream_into` (crate-private,
/// returning nothing) into a write-only sink. Positive control: the scan
/// sees a returning method when one exists.
#[test]
fn fd_proof_trace_exposes_no_reader() {
    let text = code_only(&read(&crate_root().join("src/certify/fd_proof.rs")));
    let decl = format!("pub struct {} {{", "FdProofTrace");
    let at = text.find(decl.as_str()).expect("the FdProofTrace struct");
    let body_end = at + text[at..].find("\n}").expect("struct body end");
    let body = &text[at + decl.len()..body_end];
    assert!(
        !body.contains("pub "),
        "FdProofTrace fields must be private: {body}"
    );
    let imp = format!("impl {} {{", "FdProofTrace");
    let start = text.find(imp.as_str()).expect("impl FdProofTrace");
    let mut depth = 0usize;
    let mut end = None;
    for (i, c) in text[start..].char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    end = Some(start + i);
                    break;
                }
            }
            _ => {}
        }
    }
    let block = &text[start..end.expect("impl end")];
    let check = |block: &str| -> Vec<String> {
        let mut bad = Vec::new();
        let mut methods = 0usize;
        let mut rest = block;
        while let Some(at) = rest.find("fn ") {
            let sig_end = rest[at..].find('{').map_or(rest.len(), |e| at + e);
            let sig = &rest[at..sig_end];
            let head = &rest[at.saturating_sub(16)..at];
            methods += 1;
            if let Some(ret) = sig.split("->").nth(1) {
                let ret = ret.trim();
                // The crate-private constructor returns `Self`; every other
                // method returns nothing, a class index or a flag.
                let ok = matches!(ret, "usize" | "bool")
                    || (ret == "Self" && head.contains("pub(crate)"));
                if !ok {
                    bad.push(sig.to_string());
                }
            }
            rest = &rest[sig_end..];
        }
        if methods < 4 {
            bad.push(format!("only {methods} methods found: the scan is VOID"));
        }
        bad
    };
    let bad = check(block);
    assert!(bad.is_empty(), "FdProofTrace exposes a reader: {bad:?}");
    for private in ["fn new(", "fn stream_into("] {
        let at = block.find(private).expect("method present");
        let head = &block[at.saturating_sub(12)..at];
        assert!(
            head.contains("pub(crate)"),
            "`{private}` must stay crate-private"
        );
    }
    let probe = format!(
        "impl X {{ {p}(crate) fn new() -> Self {{}} {p} fn a(&self) -> usize {{}} \
         {p} fn b(&self) -> bool {{}} fn c(self) {{}} {p} fn leaks(&self) -> &[u8] {{}} }}",
        p = "pub"
    );
    assert_eq!(
        check(&probe).len(),
        1,
        "positive control: the scan sees a reader"
    );
}
