#!/usr/bin/env bash
# run_tests.sh — Run all tests (unit + MPI) in the dev environment.
#
# Assumes that './dev_setup.sh --data' has been called at least once.
#
# Usage:
#   ./run_tests.sh              # run all tests
#   ./run_tests.sh unit         # run only unit tests
#   ./run_tests.sh mpi          # run only MPI tests

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
cd "$SCRIPT_DIR"

SUITE="all"

for arg in "$@"; do
    case $arg in
        unit)     SUITE="unit" ;;
        mpi)      SUITE="mpi" ;;
        all)      SUITE="all" ;;
        ci)       SUITE="ci" ;;
        -h|--help)
            echo "Usage: ./run_tests.sh [unit|mpi|all|ci]"
            echo ""
            echo "  unit      Run only unit tests"
            echo "  mpi       Run only MPI tests"
            echo "  all       Run all tests (default)"
            echo "  ci        Run only the tests that CI runs (skip slow/data tests)"
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
if [ -z "${VIRTUAL_ENV:-}" ]; then
    echo "Error: Environment not active. Run 'source env.sh' first."
    echo "(If you haven't set up yet, run './dev_setup.sh' first.)"
    exit 1
fi

FAILED=0

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
                     tests/unit-mpi/test_h5py.py \
                     tests/unit-mpi/test_positions.py \
                     tests/unit-mpi/test_single_cell_positions.py \
                     tests/unit-mpi/test_single_cell_write_weights.py \
                     tests/unit-mpi/test_single_cell_write_weights_distant.py; do
        echo ""
        echo "--- mpirun -n 2: $test_file ---"
        mpirun -n 2 python -m pytest "$test_file" --with-mpi -v || FAILED=1
    done
fi

# -------------------------
# CI-like tests (mirrors GitHub Actions)
# -------------------------
if [[ "$SUITE" == "ci" ]]; then
    echo ""
    echo "========================================="
    echo "  Running CI tests (skip_in_ci excluded)"
    echo "========================================="
    python -m pytest tests/unit/ -v -m "not skip_in_ci" || FAILED=1

    echo ""
    echo "========================================="
    echo "  Running CI MPI tests (skip_in_ci excluded)"
    echo "========================================="
    mpirun -n 2 python -m pytest tests/unit-mpi/ --with-mpi -v -m "not skip_in_ci" || FAILED=1
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
