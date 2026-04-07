#!/usr/bin/env bash
set -euo pipefail

# Usage:
# ./scripts/build_qlib_rdagent_v1.sh [--cuda <cuda_tag>]
# Examples:
#  CPU build (default): ./scripts/build_qlib_rdagent_v1.sh
#  GPU build with CUDA 12.1 wheels: ./scripts/build_qlib_rdagent_v1.sh --cuda 121

TAG=qlib-rdagent:v1
DOCKERFILE=Dockerfile.v1
USE_CUDA=0
TORCH_CUDA=cpu

while [[ $# -gt 0 ]]; do
	case "$1" in
		--cuda)
			shift
			if [[ -z "${1-}" ]]; then
				echo "Missing CUDA tag after --cuda" >&2; exit 1
			fi
			USE_CUDA=1
			TORCH_CUDA="$1"
			shift
			;;
		--tag)
			shift
			TAG="$1"
			shift
			;;
		-h|--help)
			sed -n '1,120p' "$0"
			exit 0
			;;
		*)
			echo "Unknown arg: $1" >&2; exit 1
			;;
	esac
done

BUILD_ARGS=(--build-arg "USE_CUDA=${USE_CUDA}" --build-arg "TORCH_CUDA=${TORCH_CUDA}")

echo "Building ${TAG} using ${DOCKERFILE} (USE_CUDA=${USE_CUDA}, TORCH_CUDA=${TORCH_CUDA})"
docker build -t "${TAG}" -f "${DOCKERFILE}" "${BUILD_ARGS[@]}" .

echo "Built image: ${TAG}"

