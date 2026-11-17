// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use std::time::Duration;

fn tiny() -> State<f32> {
    State::packed(
        PackedRows {
            n_in: 8,
            p: 1,
            coordinates: Array2::from_shape_vec((1, 1), vec![2]).unwrap(),
            m: Array2::from_shape_vec((1, 2), vec![1.25, -0.5]).unwrap(),
            d: Array2::from_shape_vec((1, 2), vec![-0.125, 0.0625]).unwrap(),
        },
        vec![0.25],
        vec![0x104],
    )
}
fn rows(s: &State<f32>) -> &PackedRows<f32> {
    let ReluRows::Packed(r) = &s.rows else {
        panic!("expected packed")
    };
    r
}
fn ids(s: &State<f32>) -> [usize; 5] {
    let r = rows(s);
    [
        r.coordinates.as_ptr() as usize,
        r.m.as_ptr() as usize,
        r.d.as_ptr() as usize,
        s.gerr.as_ptr() as usize,
        s.support.as_ptr() as usize,
    ]
}
fn unchanged(s: &State<f32>) {
    let r = rows(s);
    assert_eq!(r.coordinates.as_slice().unwrap(), &[2]);
    assert_eq!(r.m.as_slice().unwrap(), &[1.25, -0.5]);
    assert_eq!(r.d.as_slice().unwrap(), &[-0.125, 0.0625]);
    assert_eq!(s.gerr, [0.25]);
    assert_eq!(s.support, [0x104]);
}

#[test]
fn shared_take_rows_isolates_clones_and_last_use_moves() {
    let mut state = vec![Some(tiny())];
    let mut uses = vec![2];
    let original = ids(state[0].as_ref().unwrap());
    let mut owned = take_rows(&mut state, &mut uses, 0).unwrap();
    assert_eq!(uses, [1]);
    let ReluRows::Packed(r) = &mut owned.rows else {
        panic!("packed")
    };
    r.coordinates[[0, 0]] = 3;
    r.m[[0, 0]] = 99.0;
    r.d[[0, 0]] = 88.0;
    owned.gerr[0] = 77.0;
    owned.support[0] = 0;
    unchanged(state[0].as_ref().unwrap());
    assert_eq!(ids(state[0].as_ref().unwrap()), original);
    let last = take_rows(&mut state, &mut uses, 0).unwrap();
    unchanged(&last);
    assert_eq!(ids(&last), original, "last use must move every allocation");
    assert_eq!(uses, [0]);
    assert!(state[0].is_none());
}

#[test]
fn alias_add_releases_two_edges_but_preserves_later_fanout() {
    let mut state = vec![Some(tiny())];
    let mut uses = vec![3];
    let original = ids(state[0].as_ref().unwrap());
    let (sum, _) = {
        let a = borrow(&state, &uses, 0).unwrap();
        let b = borrow(&state, &uses, 0).unwrap();
        assert!(std::ptr::eq(a, b));
        rows(a)
            .add_packed(rows(b), &[1.0; 9], None)
            .unwrap()
            .expect("40B compact result fits 72B dense pair")
    };
    assert_eq!(sum.m[[0, 0]], 2.5);
    assert_eq!(sum.m[[0, 1]], -1.0);
    release(&mut state, &mut uses, 0).unwrap();
    assert_eq!(uses, [2]);
    unchanged(state[0].as_ref().unwrap());
    release(&mut state, &mut uses, 0).unwrap();
    assert_eq!(uses, [1]);
    unchanged(state[0].as_ref().unwrap());
    assert_eq!(ids(state[0].as_ref().unwrap()), original);
    let later = take_rows(&mut state, &mut uses, 0).unwrap();
    unchanged(&later);
    assert_eq!(ids(&later), original);
    assert_eq!(uses, [0]);
    assert!(state[0].is_none());
    assert!(borrow(&state, &uses, 0).is_err());
    assert!(release(&mut state, &mut uses, 0).is_err());
    assert_eq!(uses, [0]);
}

#[test]
fn take_dense_deadline_preserves_source_and_final_dense_use_moves() {
    for initial in [1usize, 2] {
        let mut state = vec![Some(tiny())];
        let mut uses = vec![initial];
        let original = ids(state[0].as_ref().unwrap());
        let expired = Instant::now().checked_sub(Duration::from_secs(1)).unwrap();
        assert!(matches!(
            take_dense(&mut state, &mut uses, 0, Some(expired)),
            Err(NyError::DeadlineExceeded(_))
        ));
        assert_eq!(uses, [initial]);
        unchanged(state[0].as_ref().unwrap());
        assert_eq!(ids(state[0].as_ref().unwrap()), original);
        let (mut m, mut d, mut g, mut support) =
            take_dense(&mut state, &mut uses, 0, None).unwrap();
        assert_eq!(m.dim(), (1, 9));
        assert_eq!(d.dim(), (1, 9));
        assert_eq!((m[[0, 2]], m[[0, 8]]), (1.25, -0.5));
        assert_eq!((d[[0, 2]], d[[0, 8]]), (-0.125, 0.0625));
        for i in (0..9).filter(|&i| i != 2 && i != 8) {
            assert_eq!(m[[0, i]].to_bits(), 0);
            assert_eq!(d[[0, i]].to_bits(), 0);
        }
        assert_eq!(g, [0.25]);
        assert_eq!(support, [0x104]);
        assert_eq!(uses, [initial - 1]);
        if initial == 1 {
            assert!(state[0].is_none());
            assert_eq!(g.as_ptr() as usize, original[3]);
            assert_eq!(support.as_ptr() as usize, original[4]);
        } else {
            m[[0, 2]] = 99.0;
            d[[0, 2]] = 88.0;
            g[0] = 77.0;
            support[0] = 0;
            unchanged(state[0].as_ref().unwrap());
            assert_eq!(ids(state[0].as_ref().unwrap()), original);
        }
    }
    let seed = tiny();
    let (m, d) = rows(&seed).materialize();
    let mut state = vec![Some(State::dense(m, d, seed.gerr, seed.support))];
    let mut uses = vec![1];
    let original = {
        let s = state[0].as_ref().unwrap();
        let ReluRows::Dense(m, d) = &s.rows else {
            panic!("dense")
        };
        [
            m.as_ptr() as usize,
            d.as_ptr() as usize,
            s.gerr.as_ptr() as usize,
            s.support.as_ptr() as usize,
        ]
    };
    let (m, d, g, support) = take_dense(&mut state, &mut uses, 0, None).unwrap();
    assert_eq!(
        [
            m.as_ptr() as usize,
            d.as_ptr() as usize,
            g.as_ptr() as usize,
            support.as_ptr() as usize
        ],
        original
    );
    assert_eq!(uses, [0]);
    assert!(state[0].is_none());
}
