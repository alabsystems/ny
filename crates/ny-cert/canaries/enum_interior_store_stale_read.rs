//! R186 CANARY (stale-read false-proof class, store side): a store through a
//! projected walk into an enum variant's interior. The bridge lowers the walk
//! as the walked pointer at an UNDEFINED byte offset (`GEP` with an `Undef`
//! index — provenance kept, offset unknown). A model that leaves the store
//! unmodeled while the enum's cell keeps its precise pre-store value — the
//! severed-provenance draft the deref-store pins caught — lets
//! `stale_after_variant_store`'s assert falsely PROVE that the field kept its
//! pre-store value. The required verdicts, in every world:
//!
//! * `stale_after_variant_store`: the `v == 1` assert must NEVER prove
//!   (the store wrote 7 through the walk). unknown/refuted are both sound.
//! * `untouched_neighbour_cell`: the `keep == 9` assert SHOULD prove under
//!   whole-cell invalidation (`keep` is a different cell the store cannot
//!   reach), and MAY stay unknown under a fail-closed refusal — but must
//!   never fail.
//!
//! Run: rustc -Ztrust-verify=on --crate-type lib <this file>

pub enum Slot {
    A(i64),
    B(i64),
}

pub fn stale_after_variant_store() -> i64 {
    let mut e = Slot::A(1);
    if let Slot::A(field) = &mut e {
        *field = 7;
    }
    match e {
        Slot::A(v) => {
            assert!(v == 1);
            v
        }
        Slot::B(v) => v,
    }
}

pub fn untouched_neighbour_cell() -> i64 {
    let mut e = Slot::A(1);
    let keep: i64 = 9;
    if let Slot::A(field) = &mut e {
        *field = 7;
    }
    assert!(keep == 9);
    keep
}
