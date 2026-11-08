#!/usr/bin/env bash
# Replay tracking and posthoc evaluation using existing cached inputs.
set -euo pipefail
if [[ $# -ne 3 ]]; then
  echo "Usage: $0 B_no_ground|C_mesh 50|300 NEW_OUTPUT_DIRECTORY" >&2
  exit 2
fi
mode="$1"
frames="$2"
case "$mode" in B_no_ground|C_mesh) ;; *) exit 2 ;; esac
case "$frames" in 50) input_name=full_input ;; 300) input_name=full_input30 ;; *) exit 2 ;; esac
project_root="${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
data_root="${DATA_ROOT:-$project_root/datasets/scout/extracted/scout}"
input_root="${INPUT_ROOT:-$project_root/outputs/posetrack-mesh-20261104/$input_name}"
image_name="${IMAGE_NAME:-scout-mtmc/posetrack-all:ba727e6}"
[[ -f "$input_root/observations.json" && -f "$data_root/meshes/mesh_ground_only.ply" ]]
if [[ -e "$3" ]]; then echo "Output already exists; choose a new directory." >&2; exit 2; fi
mkdir -p -- "$3"
output_root="$(cd -- "$3" && pwd)"
common=(--rm --runtime=runc --network=none -e NVIDIA_VISIBLE_DEVICES=void
  -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=1 -e PYTHONDONTWRITEBYTECODE=1
  --cpus=4 --memory=16g --user "$(id -u):$(id -g)"
  -v "$project_root:/project:ro" -v "$data_root:/data:ro"
  -v "$input_root:/out/input:ro" -v "$output_root:/out/replay"
  -v "$project_root/vendor:/vendor:ro")
docker run "${common[@]}" "$image_name" python -u /project/scripts/run-posetrack-mesh.py \
  --mode "$mode" --input /out/input --out /out/replay --frames "$frames" --solver highs \
  > "$output_root/run.log" 2>&1
docker run "${common[@]}" "$image_name" python /project/scripts/evaluate-posetrack-mesh.py \
  --root /out --run replay --input input --frames "$frames" > "$output_root/evaluation.log" 2>&1
printf 'Tracking and evaluation finished: %s\n' "$output_root"
