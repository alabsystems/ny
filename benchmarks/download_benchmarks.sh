#!/bin/bash
# Copyright 2026 Andrew Yates.
# Author: Andrew Yates
# Licensed under the Apache License, Version 2.0

# Download VNN-COMP benchmark repos (default: 2021, 2023-2026; 2022 optional)
#
# Usage: ./download_benchmarks.sh [years...]
# Examples:
#   ./download_benchmarks.sh           # Download default years: 2021, 2023, 2024, 2025, 2026
#   ./download_benchmarks.sh 2021      # Download only 2021
#   ./download_benchmarks.sh 2023 2024 # Download 2023 and 2024

set -e

cd "$(dirname "$0")"

write_vnncomp2025_wget_shim() {
    local compat_dir="$1"

    cat > "$compat_dir/wget" <<'EOF'
#!/bin/sh
set -e

out=""
url=""
while [ "$#" -gt 0 ]; do
    case "$1" in
        -O)
            out="$2"
            shift 2
            ;;
        --output-document=*)
            out="${1#*=}"
            shift
            ;;
        http://*|https://*)
            url="$1"
            shift
            ;;
        *)
            shift
            ;;
    esac
done

if [ -z "$url" ] || [ -z "$out" ]; then
    echo "wget compatibility shim only supports URL -O OUTPUT" >&2
    exit 1
fi

exec curl -L "$url" -o "$out"
EOF
    chmod +x "$compat_dir/wget"
}

write_vnncomp2025_gunzip_shim() {
    local compat_dir="$1"
    local system_gunzip="$2"

    cat > "$compat_dir/gunzip" <<EOF
#!/bin/sh
set -eu

real_gunzip='$system_gunzip'
stderr_file=\$(mktemp)

if "\$real_gunzip" "\$@" 2>"\$stderr_file"; then
    cat "\$stderr_file" >&2
    rm -f "\$stderr_file"
    exit 0
fi

status=\$?
cat "\$stderr_file" >&2

if grep -Ev '(^[[:space:]]*$|already exists -- skipping|unknown suffix -- ignored)' "\$stderr_file" >/dev/null 2>&1; then
    rm -f "\$stderr_file"
    exit "\$status"
fi

rm -f "\$stderr_file"
exit 0
EOF
    chmod +x "$compat_dir/gunzip"
}

run_vnncomp2025_setup() {
    local dir="$1"
    local compat_dir
    local system_gunzip
    local archive_url="https://rwth-aachen.sciebo.de/s/RapAoed1dxG1PMs/download"

    system_gunzip="$(command -v gunzip)"
    compat_dir=".ny-setup-bin"

    if ! command -v wget >/dev/null 2>&1 && ! command -v curl >/dev/null 2>&1; then
        echo "[2025] setup.sh requires wget or curl"
        return 1
    fi

    (
        cd "$dir"
        mkdir -p "$compat_dir"
        if ! command -v wget >/dev/null 2>&1; then
            echo "[2025] wget not found; providing curl-backed compatibility shim for setup.sh"
            write_vnncomp2025_wget_shim "$compat_dir"
        fi

        write_vnncomp2025_gunzip_shim "$compat_dir" "$system_gunzip"
        if [ ! -d large_models ]; then
            if [ ! -f large_models.zip ]; then
                PATH="$PWD/$compat_dir:$PATH" wget "$archive_url" -O large_models.zip
            fi
            unzip -o large_models.zip -d large_models
        fi

        echo "Moving large benchmark files"
        for benchmark_dir in large_models/vnncomp2024/*
        do
            [ -d "$benchmark_dir" ] || continue

            benchmark=$(basename "$benchmark_dir")
            seed_dir="$benchmark_dir/seed_896832480"
            if [ ! -d "$seed_dir" ]; then
                seed_dir=""
                for candidate in "$benchmark_dir"/seed_*
                do
                    if [ -d "$candidate" ]; then
                        seed_dir="$candidate"
                        break
                    fi
                done
            fi
            [ -n "$seed_dir" ] || continue

            echo "Moving $benchmark from $(basename "$seed_dir")"
            find "$seed_dir" -type f | while read -r source_file
            do
                relative_path=${source_file#"$seed_dir"/}
                target_file="benchmarks/$benchmark/$relative_path"
                mkdir -p "$(dirname "$target_file")"
                mv "$source_file" "$target_file"
            done
        done

        rm -rf large_models large_models.zip

        echo "Unzipping"
        PATH="$PWD/$compat_dir:$PATH" gunzip -r benchmarks/

        echo "CREATING HARDCODED SYMLINKS FOR BROKEN BENCHMARKS"
        # `ln` fails outright when the link's parent directory is absent, and a
        # benchmark whose own archive did not land has no `onnx/` directory. That
        # failure used to escape as the whole download's exit status, so a single
        # missing optional benchmark reported "benchmark download failed" and hid
        # every category that HAD been fetched. Create the parent first, and let a
        # link whose source is genuinely missing warn instead of aborting the run.
        link_broken_benchmark() {
            local target="$1" link="$2"
            mkdir -p "$(dirname "$link")"
            if ! ln -sf "$target" "$link"; then
                echo "WARNING: could not link $link -> $target (benchmark not fetched?)" >&2
            fi
        }
        link_broken_benchmark ../../nn4sys_2023/onnx/mscn_2048d.onnx benchmarks/nn4sys/onnx/mscn_2048d.onnx
        link_broken_benchmark ../../nn4sys_2023/onnx/mscn_2048d_dual.onnx benchmarks/nn4sys/onnx/mscn_2048d_dual.onnx
        link_broken_benchmark ../../vggnet16_2023/onnx/vgg16-7.onnx benchmarks/vggnet16_2022/onnx/vgg16-7.onnx

        rm -f "$compat_dir/wget" "$compat_dir/gunzip"
        rmdir "$compat_dir"
    )
}

run_vnncomp_setup() {
    local year="$1"
    local dir="$2"
    local compat_dir
    local system_gunzip

    if [ "$year" = "2025" ]; then
        run_vnncomp2025_setup "$dir"
    else
        (
            cd "$dir"
            system_gunzip="$(command -v gunzip)"
            compat_dir=".ny-setup-bin"
            mkdir -p "$compat_dir"
            write_vnncomp2025_gunzip_shim "$compat_dir" "$system_gunzip"
            PATH="$PWD/$compat_dir:$PATH" ./setup.sh
            rm -f "$compat_dir/gunzip"
            rmdir "$compat_dir"
        )
    fi
}

vnncomp2025_setup_targets_need_refresh() {
    local dir="$1"
    local symlink_target
    local file_target="$dir/benchmarks/cgan_2023/onnx/cGAN_imgSz32_nCh_3_small_transformer.onnx"

    for symlink_target in \
        "$dir/benchmarks/nn4sys/onnx/mscn_2048d.onnx" \
        "$dir/benchmarks/nn4sys/onnx/mscn_2048d_dual.onnx" \
        "$dir/benchmarks/vggnet16_2022/onnx/vgg16-7.onnx"
    do
        if [ ! -L "$symlink_target" ] || [ ! -e "$symlink_target" ]; then
            return 0
        fi
    done

    # setup.sh also materializes select benchmark assets from the downloaded
    # large-model archive. Refresh existing 2025 checkouts when those files
    # are missing even if the historical symlink repairs are already present.
    if [ ! -e "$file_target" ]; then
        return 0
    fi

    return 1
}

vnncomp2026_setup_targets_need_refresh() {
    local dir="$1"
    local file_target

    for file_target in \
        "$dir/benchmarks/vggnet16_2022/1.0/onnx/vgg16-7.onnx" \
        "$dir/benchmarks/vggnet16_2022/2.0/onnx/vgg16-7.onnx" \
        "$dir/benchmarks/cgan2026/1.0/onnx/cGAN_imgSz32_nCh_3_small_transformer.onnx" \
        "$dir/benchmarks/challenging_certified_training_2026/1.0/onnx/cifar10_eps2_wide_cnn7.onnx"
    do
        if [ ! -e "$file_target" ]; then
            return 0
        fi
    done

    return 1
}

download_year() {
    local year=$1
    local dir="vnncomp${year}"

    if [ -d "$dir" ]; then
        if [ "$year" = "2025" ] && [ -x "$dir/setup.sh" ] && vnncomp2025_setup_targets_need_refresh "$dir"; then
            echo "[$year] Existing checkout needs setup refresh for large models and broken symlinks..."
            run_vnncomp_setup "$year" "$dir"
            echo "[$year] Done"
            return
        fi

        if [ "$year" = "2026" ] && [ -x "$dir/setup.sh" ] && [ ! -f "$dir/.ny-setup-complete" ]; then
            if ! vnncomp2026_setup_targets_need_refresh "$dir"; then
                touch "$dir/.ny-setup-complete"
                echo "[$year] Already exists: $dir"
                return
            fi
            echo "[$year] Existing checkout needs upstream setup.sh..."
            run_vnncomp_setup "$year" "$dir"
            touch "$dir/.ny-setup-complete"
            echo "[$year] Done"
            return
        fi

        echo "[$year] Already exists: $dir"
        return
    fi

    echo "[$year] Downloading..."

    case $year in
        2021)
            git clone --depth 1 https://github.com/stanleybak/vnncomp2021.git "$dir"
            ;;
        2022)
            git clone --depth 1 https://github.com/stanleybak/vnncomp2022.git "$dir"
            ;;
        2023)
            git clone --depth 1 https://github.com/ChristopherBrix/vnncomp2023_benchmarks.git "$dir"
            ;;
        2024)
            git clone --depth 1 https://github.com/ChristopherBrix/vnncomp2024_benchmarks.git "$dir"
            ;;
        2025)
            git clone --depth 1 https://github.com/VNN-COMP/vnncomp2025_benchmarks.git "$dir"
            ;;
        2026)
            git clone --depth 1 https://github.com/VNN-COMP/vnncomp2026_benchmarks.git "$dir"
            ;;
        *)
            echo "Unknown year: $year"
            return 1
            ;;
    esac

    if { [ "$year" = "2025" ] || [ "$year" = "2026" ]; } && [ -x "$dir/setup.sh" ]; then
        echo "[$year] Running upstream setup.sh for large models and broken symlinks..."
        run_vnncomp_setup "$year" "$dir"
        [ "$year" = "2026" ] && touch "$dir/.ny-setup-complete"
    else
        # Decompress gzipped files
        echo "[$year] Decompressing files..."
        find "$dir" -name "*.gz" -exec gunzip -k -- {} +
    fi

    echo "[$year] Done"
}

# Default: download 2021 and 2023-2026 (exclude 2022 unless explicitly requested)
if [ "$#" -eq 0 ]; then
    YEARS=(2021 2023 2024 2025 2026)
else
    YEARS=("$@")
fi

echo "=== VNN-COMP Benchmark Downloader ==="
echo "Years: ${YEARS[*]}"
echo ""

for year in "${YEARS[@]}"; do
    download_year "$year"
done

echo ""
echo "=== Summary ==="
for dir in vnncomp*/; do
    if [ -d "$dir" ]; then
        count=$(find "$dir" -name "*.onnx" | wc -l | tr -d ' ')
        echo "$dir: $count ONNX files"
    fi
done

echo ""
echo "Run tests with: pytest -v --timeout=10"
