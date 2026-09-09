#!/usr/bin/env bash
# Keep CPU environment workers from oversubscribing the host during GPU replay.
# Explicit user settings remain authoritative.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
