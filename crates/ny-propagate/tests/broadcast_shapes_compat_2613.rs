// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ny_propagate::network;

#[ntest::timeout(10000)]
#[test]
fn test_network_broadcast_shapes_expected_outputs() {
    assert_eq!(
        network::broadcast_shapes(&[2, 3, 4], &[2, 3, 4]),
        Some(vec![2, 3, 4])
    );
    assert_eq!(
        network::broadcast_shapes(&[1, 4], &[3, 4]),
        Some(vec![3, 4])
    );
    assert_eq!(
        network::broadcast_shapes(&[4], &[2, 3, 4]),
        Some(vec![2, 3, 4])
    );
    assert_eq!(
        network::broadcast_shapes(&[1, 3, 1], &[2, 1, 4]),
        Some(vec![2, 3, 4])
    );
    assert_eq!(network::broadcast_shapes(&[], &[3, 4]), Some(vec![3, 4]));
    assert_eq!(network::broadcast_shapes(&[3, 4], &[]), Some(vec![3, 4]));
    assert_eq!(
        network::broadcast_shapes(&[0, 1, 8], &[1, 7, 8]),
        Some(vec![0, 7, 8])
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_network_broadcast_shapes_incompatible_still_returns_none() {
    assert_eq!(network::broadcast_shapes(&[3], &[4]), None);
    assert_eq!(network::broadcast_shapes(&[2, 3], &[4, 3]), None);
    assert_eq!(network::broadcast_shapes(&[2, 3, 4], &[2, 5, 4]), None);
}
