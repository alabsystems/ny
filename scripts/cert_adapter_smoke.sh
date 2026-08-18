#!/usr/bin/env bash
# Smoke test for the proof-carrying certificate adapter + competition mode.
# Verifies: (1) interactive beta-crown emits a .cert.json on a Verified verdict,
# (2) --competition-mode suppresses it, (3) the vnncomp entry point suppresses it.
set -u
NY=target/release/ny
D=benchmarks/vnncomp2026/benchmarks/sat_relu/1.0
ONNX=$D/onnx/unsat_v10_c26.onnx
VNNLIB=$D/vnnlib/unsat_v10_c26.vnnlib
CERT=$D/onnx/unsat_v10_c26.cert.json

echo "############ 1. interactive (proof ON by default) ############"
rm -f "$CERT"
$NY beta-crown "$ONNX" -p "$VNNLIB" --timeout 30 -v 2>&1 | grep -iE "verified|unsat|safe|cert:|status" | head -20
echo "--- cert sidecar present? ---"
if [ -f "$CERT" ]; then echo "YES: $CERT ($(wc -c < "$CERT") bytes)"; else echo "NO"; fi

echo ""
echo "############ 2. --competition-mode (proof OFF) ############"
rm -f "$CERT"
$NY beta-crown "$ONNX" -p "$VNNLIB" --timeout 30 --competition-mode -v 2>&1 | grep -iE "verified|unsat|cert:" | head -5
echo "--- cert sidecar present (expect NO)? ---"
if [ -f "$CERT" ]; then echo "UNEXPECTED YES: $CERT"; else echo "NO (correct)"; fi

echo ""
echo "############ 3. vnncomp entry point (competition mode) ############"
rm -f "$CERT" /tmp/ny_cert_smoke_results.txt
$NY vnncomp v1 sat_relu "$ONNX" "$VNNLIB" /tmp/ny_cert_smoke_results.txt 30 2>&1 | tail -2
echo "--- results file ---"; cat /tmp/ny_cert_smoke_results.txt 2>/dev/null
echo "--- cert sidecar present (expect NO)? ---"
if [ -f "$CERT" ]; then echo "UNEXPECTED YES"; else echo "NO (correct)"; fi
