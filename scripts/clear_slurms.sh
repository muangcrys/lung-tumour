#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEST_DIR="${PROJECT_DIR}/slurm_out"

mkdir -p "${DEST_DIR}"

find "${PROJECT_DIR}" \
	-path "${DEST_DIR}" -prune -o \
	-type f \( -name "slurm*.out" -o -name "slurm-*.out" \) -print0 |
while IFS= read -r -d '' file; do
	base="$(basename "$file")"
	target="${DEST_DIR}/${base}"

	if [[ -e "$target" ]]; then
		name="${base%.*}"
		ext="${base##*.}"
		i=1
		while [[ -e "${DEST_DIR}/${name}_${i}.${ext}" ]]; do
			((i++))
		done
		target="${DEST_DIR}/${name}_${i}.${ext}"
	fi

	mv "$file" "$target"
done

echo "Moved SLURM output files to: ${DEST_DIR}"
