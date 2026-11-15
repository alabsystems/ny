# VNN-COMP 2025 board state — 2026-09-14

Scored with `ny benchmarks score --results reports/measured --official
~/vnncomp2025_results --year 2025 --budget-year 2025 --track {regular,extended}`,
pin `e8222699a` (clean, `mip,cuda`). Budgets ENFORCED (`--budget-year`).

**The two leaderboards are separate and must never be combined.**

## REGULAR — total **1474.0** over 16 categories

| category | ny solved | field best | normalized |
|---|---:|---:|---:|
| cersyve | 12 | 12 | 100.0 |
| collins_rul_cnn_2022 | 62 | 62 | 100.0 |
| linearizenn_2024 | 60 | 60 | 100.0 |
| sat_relu | 100 | 100 | 100.0 |
| soundnessbench | 50 | 50 | 100.0 |
| tllverifybench_2023 | 32 | 32 | 100.0 |
| metaroom_2023 | 98 | 99 | 99.0 |
| cora_2024 | 151 | 153 | 98.7 |
| dist_shift_2023 | 71 | 72 | 98.6 |
| acasxu_2023 | 181 | 186 | 97.3 |
| malbeware | 144 | 150 | 96.0 |
| safenlp_2024 | 968 | 1080 | 89.6 |
| cgan_2023 | 18 | 21 | 85.7 |
| nn4sys | 161 | 194 | 83.0 |
| tinyimagenet_2024 | 112 | 175 | 64.0 |
| cifar100_2024 | 80 | 129 | 62.0 |

`ny_bad = 0`, verdict_flips = 0, over_budget = 0.

alpha-beta-CROWN scores 100.0 in all 16, i.e. 1600. **Deficit = 126.0.**

74 of that deficit is cifar100 (-38.0) + tinyimagenet (-36.0), both
proved by the CPU-f64 margin-row lane and both cost-bound rather than
bound-quality-bound. cgan is -14.3 over just 3 rows — the best per-row
value on the board.

## EXTENDED — total **491.0** over 9 categories

| category | ny solved | field best | normalized |
|---|---:|---:|---:|
| cctsdb_yolo_2023 | 39 | 39 | 100.0 |
| collins_aerospace_benchmark | 6 | 6 | 100.0 |
| traffic_signs_recognition_2023 | 44 | 44 | 100.0 |
| vggnet16_2022 | 14 | 18 | 77.8 |
| ml4acopf_2024 | 45 | 62 | 72.6 |
| relusplitter | 46 | 171 | 26.9 |
| lsnc_relu | 11 | 80 | 13.8 |
| vit_2023 | 0 | 125 | 0.0 |
| yolo_2023 | 0 | 71 | 0.0 |

`ny_bad = 0`, verdict_flips = 0, over_budget = 0.

Categories with no rows: ['vit_2023', 'yolo_2023']. Both were probed and are
capability-bound (vit 3/3 timeout at 91-94% of budget; yolo 2/3 timeout
at 99-102%), so they are real limits, not absences.

