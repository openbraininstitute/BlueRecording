#!/usr/bin/env bash
# run_tests.sh — Run all tests (unit + MPI) in the dev environment.
#
# Assumes that './dev_setup.sh' and './download_examples_data.sh' have been
# called at least once.
#
# Usage:
#   ./run_tests.sh              # run all tests
#   ./run_tests.sh unit         # run only unit tests (no MPI)
#   ./run_tests.sh integration  # run only integration tests (no MPI)
#   ./run_tests.sh mpi          # run only MPI tests
#   ./run_tests.sh ci           # like 'all' but skip tests marked skip_in_ci

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
cd "$SCRIPT_DIR"

SUITE="all"

for arg in "$@"; do
    case $arg in
        unit)          SUITE="unit" ;;
        integration)   SUITE="integration" ;;
        mpi)           SUITE="mpi" ;;
        all)           SUITE="all" ;;
        ci)            SUITE="ci" ;;
        -h|--help)
            echo "Usage: ./run_tests.sh [unit|integration|mpi|all|ci]"
            echo ""
            echo "  unit           Run only unit tests (no MPI)"
            echo "  integration    Run only integration tests (no MPI)"
            echo "  mpi            Run MPI tests (unit-mpi + integration-mpi)"
            echo "  all            Run all tests (default)"
            echo "  ci             Like 'all' but skip tests marked skip_in_ci"
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

# -------------------------
# Marker filter: ci mode skips data-heavy tests
# -------------------------
MARKER_ARGS=()
if [[ "$SUITE" == "ci" ]]; then
    MARKER_ARGS=(-m "not skip_in_ci")
fi

FAILED=0

# -------------------------
# Unit tests
# -------------------------
if [[ "$SUITE" == "all" || "$SUITE" == "ci" || "$SUITE" == "unit" ]]; then
    echo ""
    echo "========================================="
    echo "  Running unit tests"
    echo "========================================="
    python -m pytest tests/unit/ -v --forked ${MARKER_ARGS[@]+"${MARKER_ARGS[@]}"} || FAILED=1
fi

# -------------------------
# Integration tests
# -------------------------
if [[ "$SUITE" == "all" || "$SUITE" == "ci" || "$SUITE" == "unit" || "$SUITE" == "integration" ]]; then
    echo ""
    echo "========================================="
    echo "  Running integration tests"
    echo "========================================="
    python -m pytest tests/integration/ -v --forked ${MARKER_ARGS[@]+"${MARKER_ARGS[@]}"} || FAILED=1
fi

# -------------------------
# MPI unit tests
# -------------------------
if [[ "$SUITE" == "all" || "$SUITE" == "ci" || "$SUITE" == "mpi" ]]; then
    echo ""
    echo "========================================="
    echo "  Running MPI unit tests"
    echo "========================================="
    for test_file in tests/unit-mpi/test_*.py; do
        echo ""
        echo "--- mpirun -n 2: $test_file ---"
        mpirun -n 2 python -m pytest "$test_file" --with-mpi -v ${MARKER_ARGS[@]+"${MARKER_ARGS[@]}"}
        rc=$?
        if [[ $rc -ne 0 && $rc -ne 5 ]]; then FAILED=1; fi
    done
fi

# -------------------------
# MPI integration tests (one mpirun per file — NEURON global state)
# -------------------------
if [[ "$SUITE" == "all" || "$SUITE" == "ci" || "$SUITE" == "mpi" ]]; then
    echo ""
    echo "========================================="
    echo "  Running MPI integration tests"
    echo "========================================="
    for test_file in tests/integration-mpi/test_*.py; do
        # Skip files where all tests would be deselected by the marker filter
        if [[ ${#MARKER_ARGS[@]} -gt 0 ]]; then
            selected=$(python -m pytest "$test_file" --collect-only -q ${MARKER_ARGS[@]+"${MARKER_ARGS[@]}"} 2>/dev/null | grep -c "test_" || true)
            if [[ "$selected" -eq 0 ]]; then
                echo "--- skipping $test_file (no matching tests) ---"
                continue
            fi
        fi
        echo ""
        echo "--- mpirun -n 2: $test_file ---"
        mpirun -n 2 python -m pytest "$test_file" --with-mpi -v ${MARKER_ARGS[@]+"${MARKER_ARGS[@]}"} || FAILED=1
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
