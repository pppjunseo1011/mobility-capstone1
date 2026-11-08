#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 2 ]]; then echo "Usage: $0 ground-off|mesh|height-abstain NEW_OUTPUT_DIR" >&2; exit 2; fi
policy="$1";extra=()
case "$policy" in ground-off) mode=B_no_ground ;; mesh) mode=C_mesh ;; height-abstain) mode=C_mesh;extra=(--unknown-height-abstain) ;; *) exit 2 ;; esac
project_root="${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
data_root="${DATA_ROOT:-$project_root/datasets/scout/extracted/scout}"
study_root="${STUDY_ROOT:-$project_root/outputs/posetrack-time-study-20261106}"
input_root="${INPUT_ROOT:-$study_root/aligned_input}"
image_name="${IMAGE_NAME:-scout-mtmc/posetrack-all:ba727e6}"
[[ -f "$input_root/observations.json" && -f "$study_root/frame_mapping.csv" ]]
if [[ -e "$2" ]]; then echo "Choose a NEW output directory" >&2;exit 2;fi
mkdir -p -- "$2";output_root="$(cd -- "$2" && pwd)"
common=(--rm --runtime=runc --network=none -e NVIDIA_VISIBLE_DEVICES=void -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=1 -e PYTHONDONTWRITEBYTECODE=1
 --cpus=3 --memory=16g --user "$(id -u):$(id -g)" -v "$project_root:/project:ro" -v "$data_root:/data:ro" -v "$input_root:/input:ro"
 -v "$study_root:/study:ro" -v "$output_root:/result" -v "$project_root/vendor:/vendor:ro")
docker run "${common[@]}" "$image_name" python -u /project/scripts/run-posetrack-time-study.py --mode "$mode" --input /input --out /result --frames 300 --solver highs "${extra[@]}" > "$output_root/run.log" 2>&1
docker run "${common[@]}" "$image_name" python /project/scripts/evaluate-posetrack-time-study.py --native-run /result --observations /input/observations.json --frame-map /study/frame_mapping.csv --out /result/evaluation --config "$policy" > "$output_root/evaluation.log" 2>&1
printf 'Completed: %s\n' "$output_root"
