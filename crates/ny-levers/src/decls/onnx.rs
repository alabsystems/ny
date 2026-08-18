// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! ONNX/model-rewrite selectors.

use crate::{
    declare_levers, Bucket, DefaultSpec, LeverDecl, LeverKind, MoatRisk, Provenance, ReaderSite,
    Scope,
};

const TERMINAL_SOFTMAX: Scope = Scope {
    package: "ny-onnx",
    subsystem: "terminal-softmax-strip",
};

const BENCHMARK_CORPUS: Scope = Scope {
    package: "ny-onnx",
    subsystem: "benchmark-corpus-tests",
};

declare_levers! {
    registry ONNX_LEVERS;

    /// `NY_STRIP_TERMINAL_SOFTMAX` — staged argmax-complement rewrite.
    pub STRIP_TERMINAL_SOFTMAX = LeverDecl {
        name: "NY_STRIP_TERMINAL_SOFTMAX",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Enables the guarded terminal-Softmax strip for authenticated argmax-complement \
properties. Exact \"1\" arms it; absence and every other byte string leave the \
model and property untouched. The transform is currently staged and has no \
caller on the bound path. If wired later, every model/spec guard must still \
pass atomically before the graph output is changed. The selector is \
verdict-carrying by construction, so it remains Debug and dark until a \
retained end-to-end float-semantics qualification exists.",
        provenance: Provenance::Unmeasured {
            why_ok: "dark by default and currently has no production caller; \
                     the default path is byte-identical, while focused tests \
                     pin exact parsing, atomic refusal, and guarded rewrites",
        },
        owner: TERMINAL_SOFTMAX,
        readers: &[ReaderSite {
            scope: TERMINAL_SOFTMAX,
            role: "gate the guarded terminal-Softmax strip before any model or \
                   property inspection or mutation",
            site: "crates/ny-onnx/src/optimization.rs:strip_terminal_softmax_armed",
        }],
    };

    /// `NY_BENCH_ROOT` — external VNN-COMP 2025 corpus root.
    pub BENCH_ROOT_2025 = LeverDecl {
        name: "NY_BENCH_ROOT",
        kind: LeverKind::Text,
        default: DefaultSpec::Unset,
        bucket: Bucket::Debug,
        moat: MoatRisk::None,
        doc: "\
Overrides the external VNN-COMP 2025 benchmark root used by corpus integration \
tests. Absence walks worktree ancestors; a present value is used as a path and \
missing corpus bytes fail the test rather than silently skipping it. This \
selector cannot affect a verifier verdict or production execution.",
        provenance: Provenance::Unmeasured {
            why_ok: "test-only path selector; absence is the shipped default and \
                     a missing or invalid target fails closed instead of skipping",
        },
        owner: BENCHMARK_CORPUS,
        readers: &[ReaderSite {
            scope: BENCHMARK_CORPUS,
            role: "locate the required external VNN-COMP 2025 corpus",
            site: "crates/ny-onnx/tests/common/mod.rs:benchmark_root",
        }],
    };

    /// `NY_BENCH_ROOT_2026` — external VNN-COMP 2026 corpus root.
    pub BENCH_ROOT_2026 = LeverDecl {
        name: "NY_BENCH_ROOT_2026",
        kind: LeverKind::Text,
        default: DefaultSpec::Unset,
        bucket: Bucket::Debug,
        moat: MoatRisk::None,
        doc: "\
Overrides the external VNN-COMP 2026 benchmark root used by the terminal-Softmax \
corpus integration test. Absence walks worktree ancestors; a present value is \
used as a path and missing corpus bytes fail the test rather than silently \
skipping it. This selector cannot affect a verifier verdict or production \
execution.",
        provenance: Provenance::Unmeasured {
            why_ok: "test-only path selector; absence is the shipped default and \
                     a missing or invalid target fails closed instead of skipping",
        },
        owner: BENCHMARK_CORPUS,
        readers: &[ReaderSite {
            scope: BENCHMARK_CORPUS,
            role: "locate the required external VNN-COMP 2026 corpus",
            site: "crates/ny-onnx/tests/common/mod.rs:benchmark_root",
        }],
    };
}
