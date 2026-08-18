// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Explicit controls for long-running test and measurement expansions.

use crate::{
    declare_levers, Bucket, DefaultSpec, LeverDecl, LeverKind, MoatRisk, Provenance, ReaderSite,
    Scope,
};

const MEASUREMENT_SCOPE: Scope = Scope {
    package: "ny-test-utils",
    subsystem: "long-measurements",
};

const FULL_MEASUREMENT_READERS: &[ReaderSite] = &[
    ReaderSite {
        scope: Scope {
            package: "ny-mip",
            subsystem: "exact-star-tests",
        },
        role: "expand bounded exact-star correctness probes to their full width sweep",
        site: "crates/ny-mip/src/star_verify_tests.rs:full_measurement_mode",
    },
    ReaderSite {
        scope: Scope {
            package: "ny-mip",
            subsystem: "acasxu-star-tests",
        },
        role: "expand the bounded real-ACAS probe to the full external-corpus sweep",
        site: "crates/ny-mip/src/star_acasxu_tests.rs:acasxu_prop2_input_vs_neuron_branching",
    },
    ReaderSite {
        scope: Scope {
            package: "ny-cli",
            subsystem: "flush-charge-acceptance",
        },
        role: "run the full CPU metaroom root-pass wall-clock instead of its bounded fixture preflight",
        site: "crates/ny-cli/tests/flush_charged_metaroom_wallclock.rs:metaroom_119_cpu_root_pass_wallclock_baseline",
    },
];

declare_levers! {
    registry MEASUREMENT_LEVERS;

    /// `NY_FULL_MEASUREMENTS` — expand bounded test probes to full measurements.
    pub FULL_MEASUREMENTS = LeverDecl {
        name: "NY_FULL_MEASUREMENTS",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::None,
        doc: "\
Expands selected bounded correctness and fixture preflights into their full, \
long-running measurement forms. Exact \"1\" arms it; absence and every other \
value retain the ordinary bounded gate. It is test-only and cannot alter a \
published bound or verifier verdict, but the full forms can take minutes and \
may require external corpus data, so they remain an explicit Debug choice.",
        provenance: Provenance::Unmeasured {
            why_ok: "dark by default and test-only; the ordinary bounded forms retain assertions and the expanded work has no production verdict path",
        },
        owner: MEASUREMENT_SCOPE,
        readers: FULL_MEASUREMENT_READERS,
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{read_with, LeverValue, Source};

    #[test]
    fn full_measurements_is_default_dark_and_exact_one_only() {
        for (raw, enabled, source) in [
            (None, false, Source::Default),
            (Some("1"), true, Source::LegacyEnv),
            (Some("0"), false, Source::LegacyEnv),
            (Some("true"), false, Source::LegacyEnvRejected),
            (Some(" 1 "), false, Source::LegacyEnvRejected),
        ] {
            let resolved = read_with(&FULL_MEASUREMENTS, |_| raw.map(str::to_owned));
            assert_eq!(resolved.value, LeverValue::Bool(enabled), "{raw:?}");
            assert_eq!(resolved.source, source, "{raw:?}");
        }
    }
}
