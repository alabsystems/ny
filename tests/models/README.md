# Test Models

Small ONNX, NNet, and VNN-LIB fixtures used by the Cargo tests
(`ny_test_utils::test_models_dir()`). Keep fixtures minimal and deterministic.
Do not commit benchmark corpora: `benchmarks/download_benchmarks.sh` clones
them into the gitignored `benchmarks/vnncompYYYY/`.
