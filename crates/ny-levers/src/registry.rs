// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Per-module registry slices and the merge that turns them into one
//! workspace registry.

use std::fmt;

use crate::decl::LeverDecl;

/// The set of levers declared by one module, produced by
/// [`crate::declare_levers`].
#[derive(Debug)]
pub struct Registry {
    module: &'static str,
    decls: &'static [&'static LeverDecl],
}

impl Registry {
    /// Build a module registry. Called only by the declaration macro.
    pub const fn new(module: &'static str, decls: &'static [&'static LeverDecl]) -> Self {
        Self { module, decls }
    }

    /// The `module_path!()` the declarations live in.
    pub const fn module(&self) -> &'static str {
        self.module
    }

    /// The declarations, in declaration order.
    pub const fn decls(&self) -> &'static [&'static LeverDecl] {
        self.decls
    }
}

/// Why merging module registries failed.
///
/// Both variants are duplicate-name findings; they differ in what the author
/// has to do about it, which is why they are separate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CollectError {
    /// The same `NY_*` name is declared by two DIFFERENT `LeverDecl`s.
    ///
    /// This is the discovered-collision hazard: two subsystems independently
    /// invented the same name and now share process state without either
    /// author knowing. The fix is never "rename it in the registry" — it is to
    /// decide whether the sharing is intended. If it is, delete one
    /// declaration and add a reader site to the other (see `NY_EFT_ERR`); if
    /// it is not, rename one of the environment variables.
    NameCollision {
        /// The colliding environment variable name.
        name: &'static str,
        /// Modules that declared it.
        modules: Vec<&'static str>,
    },
    /// One `LeverDecl` was exported by two module registries, and it does NOT
    /// declare multiple reader sites.
    ///
    /// A single-reader lever belongs to exactly one module registry;
    /// re-exporting it double-counts it in the receipt. If the second export
    /// exists because a second crate really does read the variable, the fix is
    /// to add that crate's [`crate::ReaderSite`] to the declaration — which
    /// then makes this exact situation legal.
    DuplicateExport {
        /// The doubly-exported environment variable name.
        name: &'static str,
        /// Modules that exported it.
        modules: Vec<&'static str>,
    },
}

impl fmt::Display for CollectError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NameCollision { name, modules } => write!(
                f,
                "{name} is declared by two different LeverDecls ({}). Two subsystems \
                 cannot own one environment variable: either delete one declaration and \
                 add its ReaderSite to the survivor (a declared shared gate, as \
                 NY_EFT_ERR does), or rename one of the variables.",
                modules.join(", ")
            ),
            Self::DuplicateExport { name, modules } => write!(
                f,
                "{name} is exported by more than one module registry ({}) but declares \
                 fewer than two reader sites. Export it from exactly one module, or — if \
                 a second crate genuinely reads it — declare that ReaderSite so the \
                 sharing is documented rather than discovered.",
                modules.join(", ")
            ),
        }
    }
}

impl std::error::Error for CollectError {}

/// Every declared lever in the workspace, merged and name-sorted.
#[derive(Debug, Default)]
pub struct LeverRegistry {
    decls: Vec<&'static LeverDecl>,
}

impl LeverRegistry {
    /// All declarations, sorted by name.
    pub fn all(&self) -> &[&'static LeverDecl] {
        &self.decls
    }

    /// Number of distinct declared levers.
    pub fn len(&self) -> usize {
        self.decls.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.decls.is_empty()
    }

    /// Look a declaration up by environment variable name.
    pub fn get(&self, name: &str) -> Option<&'static LeverDecl> {
        self.decls
            .binary_search_by(|d| d.name.cmp(name))
            .ok()
            .map(|i| self.decls[i])
    }
}

/// Merge module registries into one workspace registry.
///
/// Duplicate-name detection distinguishes three situations:
///
/// 1. Two different declarations, one name → [`CollectError::NameCollision`].
/// 2. The SAME declaration exported twice, with fewer than two declared reader
///    sites → [`CollectError::DuplicateExport`].
/// 3. The same declaration exported twice, WITH two or more declared reader
///    sites → accepted, deduplicated to a single entry. This is the deliberate
///    cross-crate twin gate: `NY_EFT_ERR` is one lever with a `ny-propagate`
///    reader and two `ny-gpu` readers, and each reading crate's module
///    registry may legitimately name it. The sharing is legal precisely
///    BECAUSE it is written down; an undeclared shared read is case 1 or 2.
pub fn collect(registries: &[&'static Registry]) -> Result<LeverRegistry, CollectError> {
    // Small n (tens to low hundreds); a flat scan keeps the error messages
    // able to name every contributing module without a second index.
    let mut decls: Vec<&'static LeverDecl> = Vec::new();
    let mut modules: Vec<&'static str> = Vec::new();

    for reg in registries {
        for decl in reg.decls() {
            match decls.iter().position(|d| d.name == decl.name) {
                None => {
                    decls.push(decl);
                    modules.push(reg.module());
                }
                Some(idx) => {
                    let same_decl = std::ptr::eq(decls[idx], *decl);
                    let mut involved = vec![modules[idx], reg.module()];
                    involved.dedup();
                    if !same_decl {
                        return Err(CollectError::NameCollision {
                            name: decl.name,
                            modules: involved,
                        });
                    }
                    if !decl.is_multi_reader() {
                        return Err(CollectError::DuplicateExport {
                            name: decl.name,
                            modules: involved,
                        });
                    }
                    // Case 3: a declared multi-reader lever. Already recorded.
                }
            }
        }
    }

    decls.sort_unstable_by_key(|d| d.name);
    Ok(LeverRegistry { decls })
}

#[cfg(test)]
mod tests {
    use crate::decl::{
        Bucket, DefaultSpec, LeverDecl, LeverKind, MoatRisk, Provenance, ReaderSite, Scope,
    };
    use crate::registry::{collect, CollectError, Registry};

    const fn scope(package: &'static str) -> Scope {
        Scope {
            package,
            subsystem: "selftest",
        }
    }

    const ONE_READER: &[ReaderSite] = &[ReaderSite {
        scope: scope("ny-levers"),
        role: "only reader",
        site: "src/registry.rs (fixture)",
    }];

    const TWO_READERS: &[ReaderSite] = &[
        ReaderSite {
            scope: scope("ny-propagate"),
            role: "cpu arm",
            site: "src/registry.rs (fixture)",
        },
        ReaderSite {
            scope: scope("ny-gpu"),
            role: "gpu twin",
            site: "src/registry.rs (fixture)",
        },
    ];

    const fn fixture(name: &'static str, readers: &'static [ReaderSite]) -> LeverDecl {
        LeverDecl {
            name,
            kind: LeverKind::Bool,
            default: DefaultSpec::Bool(false),
            bucket: Bucket::Debug,
            moat: MoatRisk::None,
            doc: "registry unit-test fixture",
            provenance: Provenance::Unmeasured { why_ok: "test" },
            owner: scope("ny-levers"),
            readers,
        }
    }

    /// Exercises the macro exactly as a real declaration module would use it.
    mod declared {
        crate::declare_levers! {
            registry SELFTEST_LEVERS;

            /// A single-reader fixture lever.
            pub SOLO = super::fixture("NY_LEVERS_SELFTEST_SOLO", super::ONE_READER);

            /// A fixture lever with two declared reader sites.
            pub SHARED = super::fixture("NY_LEVERS_SELFTEST_SHARED", super::TWO_READERS);
        }
    }

    /// A SECOND, independent declaration that happens to reuse a name — the
    /// discovered-collision hazard.
    static TWIN_IMPOSTOR: LeverDecl = fixture("NY_LEVERS_SELFTEST_SHARED", ONE_READER);

    static SOLO_ECHO: Registry = Registry::new("test::solo_echo", &[&declared::SOLO]);
    static SHARED_ECHO: Registry = Registry::new("test::shared_echo", &[&declared::SHARED]);
    static IMPOSTOR_REG: Registry = Registry::new("test::impostor", &[&TWIN_IMPOSTOR]);

    #[test]
    fn macro_files_declarations_in_a_module_registry() {
        let reg = &declared::SELFTEST_LEVERS;
        assert!(
            reg.module().ends_with("registry::tests::declared"),
            "module_path! must resolve at the declaration site, got {}",
            reg.module()
        );
        let names: Vec<&str> = reg.decls().iter().map(|d| d.name).collect();
        assert_eq!(
            names,
            vec!["NY_LEVERS_SELFTEST_SOLO", "NY_LEVERS_SELFTEST_SHARED"],
            "the registry slice must list the module's declarations in order"
        );
        assert!(
            std::ptr::eq(reg.decls()[0], &raw const declared::SOLO),
            "the slice must point at the very statics the macro declared"
        );
    }

    #[test]
    fn collect_sorts_and_indexes_by_name() {
        let merged = collect(&[&declared::SELFTEST_LEVERS]).expect("single registry merges");
        assert_eq!(merged.len(), 2);
        assert!(!merged.is_empty());
        let names: Vec<&str> = merged.all().iter().map(|d| d.name).collect();
        assert_eq!(
            names,
            vec!["NY_LEVERS_SELFTEST_SHARED", "NY_LEVERS_SELFTEST_SOLO"]
        );
        assert!(merged.get("NY_LEVERS_SELFTEST_SOLO").is_some());
        assert!(merged.get("NY_NOT_DECLARED_ANYWHERE").is_none());
    }

    #[test]
    fn arm_one_same_decl_exported_twice_without_declared_readers_is_an_error() {
        let err = collect(&[&declared::SELFTEST_LEVERS, &SOLO_ECHO])
            .expect_err("a re-exported single-reader lever must not merge");
        match err {
            CollectError::DuplicateExport { name, ref modules } => {
                assert_eq!(name, "NY_LEVERS_SELFTEST_SOLO");
                assert_eq!(modules.len(), 2, "the error must name both modules");
            }
            other => panic!("expected DuplicateExport, got {other:?}"),
        }
        assert!(
            err.to_string().contains("declare that ReaderSite"),
            "the message must point at the fix, got: {err}"
        );
    }

    #[test]
    fn arm_two_declared_multi_reader_lever_may_be_exported_by_every_reader() {
        // The NY_EFT_ERR shape: ONE declaration, reader sites written down,
        // legitimately named by each reading crate's module registry.
        let merged = collect(&[&SHARED_ECHO, &SHARED_ECHO])
            .expect("a declared multi-reader lever may be exported by every reader's module");
        assert_eq!(merged.len(), 1, "it must merge to exactly one entry");
        assert_eq!(merged.all()[0].name, "NY_LEVERS_SELFTEST_SHARED");
        assert_eq!(
            merged.all()[0].reader_packages(),
            vec!["ny-propagate", "ny-gpu"]
        );
    }

    #[test]
    fn two_different_decls_with_one_name_collide() {
        let err = collect(&[&SHARED_ECHO, &IMPOSTOR_REG])
            .expect_err("independent declarations must not share a name");
        match err {
            CollectError::NameCollision { name, .. } => {
                assert_eq!(name, "NY_LEVERS_SELFTEST_SHARED");
            }
            other => panic!("expected NameCollision, got {other:?}"),
        }
        assert!(
            err.to_string().contains("rename one of the variables"),
            "the message must offer both fixes, got: {err}"
        );
    }

    #[test]
    fn is_multi_reader_is_the_predicate_that_separates_the_arms() {
        assert!(!declared::SOLO.is_multi_reader());
        assert!(declared::SHARED.is_multi_reader());
    }
}
