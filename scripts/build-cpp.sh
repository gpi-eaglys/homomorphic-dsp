#! /usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR=$(realpath "${SCRIPT_DIR}/..")
BLD_DIR_REL="${REPO_DIR}/build/cmake/cmake-build-release"
BLD_DIR_DBG="${REPO_DIR}/build/cmake/cmake-build-debug"

cmake -B "${BLD_DIR_REL}" -S "${REPO_DIR}/lib/cpp" -DCMAKE_BUILD_TYPE=Release \
    -DWITH_NATIVEOPT=ON \
    -DWITH_OPENMP=ON

cmake --build "${BLD_DIR_REL}" --parallel 7

#cmake -B "${BLD_DIR_DBG}" -S "${REPO_DIR}/lib/cpp" -DCMAKE_BUILD_TYPE=Debug
#cmake --build "${BLD_DIR_DBG}" --parallel
