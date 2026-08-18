// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Official VNN-COMP 2025 track membership.
//!
//! These lists are pinned from `REGULAR_BENCHMARKS` and
//! `EXTENDED_BENCHMARKS` in both released scorer configurations
//! (`SCORING-SMALL-TOL/settings.py` and `SCORING-ZERO-TOL/settings.py`) at
//! VNN-COMP/vnncomp2025_results commit
//! `ea89fbc2518b6729f17c96eeec22c56c88e496a9`.  The released scorer spells
//! category ids with a `2025_` prefix internally; result CSVs and NY use the
//! unprefixed ids below.

pub(crate) const REGULAR_TRACK_2025: [&str; 16] = [
    "acasxu_2023",
    "cersyve",
    "cgan_2023",
    "cifar100_2024",
    "collins_rul_cnn_2022",
    "cora_2024",
    "dist_shift_2023",
    "linearizenn_2024",
    "malbeware",
    "metaroom_2023",
    "nn4sys",
    "safenlp_2024",
    "sat_relu",
    "soundnessbench",
    "tinyimagenet_2024",
    "tllverifybench_2023",
];

pub(crate) const EXTENDED_TRACK_2025: [&str; 10] = [
    "cctsdb_yolo_2023",
    "collins_aerospace_benchmark",
    "lsnc_relu",
    "ml4acopf_2023",
    "ml4acopf_2024",
    "relusplitter",
    "traffic_signs_recognition_2023",
    "vggnet16_2022",
    "vit_2023",
    "yolo_2023",
];
