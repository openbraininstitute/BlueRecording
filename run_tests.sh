#!/usr/bin/env bash
# run_tests.sh — Run all tests (unit + MPI) in the dev environment.
#
# Assumes that 'source setup.sh --dev --data' has been called at least once
# and the virtual environment is active (i.e. you are in the venv).
#
# Usage:
#   ./run_tests.sh              # run all tests
#   ./run_tests.sh unit         # run only unit tests
#   ./run_tests.sh mpi          # run only MPI tests
#   ./run_tests.sh --setup      # re-run setup before testing

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
cd "$SCRIPT_DIR"

RUN_SETUP=0
SUITE="all"

for arg in "$@"; do
    case $arg in
        --setup)  RUN_SETUP=1 ;;
        unit)     SUITE="unit" ;;
        mpi)      SUITE="mpi" ;;
        all)      SUITE="all" ;;
        -h|--help)
            echo "Usage: ./run_tests.sh [--setup] [unit|mpi|all]"
            echo ""
            echo "  --setup   Source setup.sh --dev --data before running tests"
            echo "  unit      Run only unit tests"
            echo "  mpi       Run only MPI tests"
            echo "  all       Run all tests (default)"
            exit 0
            ;;
        *)
            echo "Unknown option: $arg (try --help)"
            exit 1
            ;;
    esac
done

# -------------------------
# Environment setup
# -------------------------
if [[ "$RUN_SETUP" -eq 1 ]]; then
    echo "=== Running setup.sh --dev --data ==="
    source setup.sh --dev --data
elif [[ -d "venv" ]]; then
    source venv/bin/activate
else
    echo "Error: No venv found. Run with --setup first, or 'source setup.sh --dev --data' manually."
    exit 1
fi

FAILED=0

echo ""
echo "Note: This script assumes 'source setup.sh --dev --data' was called at least once"
echo "      and the virtual environment is active."
echo ""

# -------------------------
# Unit tests
# -------------------------
if [[ "$SUITE" == "all" || "$SUITE" == "unit" ]]; then
    echo ""
    echo "========================================="
    echo "  Running unit tests"
    echo "========================================="
    python -m pytest tests/unit/ -v --forked || FAILED=1
fi

# -------------------------
# MPI tests
# -------------------------
if [[ "$SUITE" == "all" || "$SUITE" == "mpi" ]]; then
    echo ""
    echo "========================================="
    echo "  Running MPI tests"
    echo "========================================="
    for test_file in tests/unit-mpi/test_write_weights.py \
                     tests/unit-mpi/test_h5py_MPI.py \
                     tests/unit-mpi/test_get_positions.py; do
        echo ""
        echo "--- mpirun -n 2: $test_file ---"
        mpirun -n 2 python -m pytest "$test_file" --with-mpi -v || FAILED=1
    done
fi

# -------------------------
# Summary
# -------------------------
echo ""
if [[ "$FAILED" -eq 1 ]]; then
    echo "⚠  Some tests failed."
    exit 1
else
    echo "✅ All tests passed."
fi
