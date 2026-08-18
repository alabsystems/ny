// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! The declaration macro.

/// Declare a module's levers and its registry slice in one place.
///
/// The macro emits one `static` per lever plus a `static` [`crate::Registry`]
/// listing them, so a declaration cannot exist without being filed: the two
/// halves are written by the same expansion and there is no way to add a
/// `LeverDecl` to the module and forget the registry entry.
///
/// ```
/// use ny_levers::{
///     declare_levers, Bucket, DefaultSpec, LeverDecl, LeverKind, MoatRisk, Provenance,
///     ReaderSite, Scope,
/// };
///
/// declare_levers! {
///     registry EXAMPLE_LEVERS;
///
///     /// Doc comments on the handle are encouraged; `doc` is for the value.
///     pub EXAMPLE = LeverDecl {
///         name: "NY_EXAMPLE",
///         kind: LeverKind::Bool,
///         default: DefaultSpec::Bool(false),
///         bucket: Bucket::Debug,
///         moat: MoatRisk::None,
///         doc: "what it does when set",
///         provenance: Provenance::Unmeasured { why_ok: "dark by default" },
///         owner: Scope { package: "ny-levers", subsystem: "docs" },
///         readers: &[],
///     };
/// }
///
/// assert_eq!(EXAMPLE_LEVERS.decls().len(), 1);
/// assert_eq!(EXAMPLE.name, "NY_EXAMPLE");
/// ```
///
/// `module_path!()` is expanded at the call site, so the registry records the
/// module the declarations actually live in.
#[macro_export]
macro_rules! declare_levers {
    (
        $(#[$rmeta:meta])*
        registry $registry:ident;
        $(
            $(#[$dmeta:meta])*
            $vis:vis $handle:ident = $decl:expr;
        )+
    ) => {
        $(
            $(#[$dmeta])*
            $vis static $handle: $crate::LeverDecl = $decl;
        )+

        $(#[$rmeta])*
        /// Every lever declared by this module, in declaration order.
        pub static $registry: $crate::Registry =
            $crate::Registry::new(module_path!(), &[$( &$handle ),+]);
    };
}
