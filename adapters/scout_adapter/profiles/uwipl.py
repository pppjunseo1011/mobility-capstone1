"""SCOUT export contract for the pinned UWIPL source; no model execution or GT reuse."""

import csv
from decimal import Decimal, InvalidOperation
import math
from pathlib import Path


SOURCE_COMMIT = "8e23a69de57d2547b774399ae45792f83a2d17ef"
FRAME_BASE = 0
FORMATS = {"aic9": 9, "mot10": 10, "mot16": 16}


def _model_id(value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("UWIPL model_camera_id must be a positive integer")
    return value


def build_profile(manifest: dict) -> dict:
    """Describe an isolated export tree, preserving upstream adaptation blockers.

    S001/cNNN are model-side aliases, not an assertion that SCOUT has AI City
    geometry. Frame files are numbered by the parent's frame map, starting at 0.
    This function neither writes files nor constructs calibration homographies.
    """
    scene_name = "S001"
    cameras = []
    directories = {
        "configs", "data/test/S001", "data/test_det", "data/test_emb",
        "data/SCT", "data/tracklet", "data/hungarian_cluster",
        "data/hungarian_cluster_S001", "pose", "pose/tracking_result",
        "pose/feet", "reproduce",
    }
    blockers = [
        "실제 추론 준비 미완료: 이 프로파일은 입력 배치와 결과 형식 계약만 제공하며 원본 실행 코드의 SCOUT 전용 수정이 필요합니다.",
        "S001/c001 등의 이름은 export 별칭입니다. AI City의 보정 행렬·카메라 가중치·ROI·scene별 threshold를 SCOUT에 재사용하면 안 됩니다.",
        "실사 검출기는 1920×1080 및 min(800/1080,1440/1920) scale을 하드코딩합니다. 실제 이미지 크기와 모델 resize ratio를 사용하도록 수정해야 합니다.",
        "Synthetic 검출기는 /data/test 입력, CWD 기준 data/test_det 출력, JPG basename 및 append 모드를 고정합니다. 경로 설정과 재실행 중복 방지가 필요합니다.",
        "ReID 추출기는 argparse 재호출, deep_person_reid 경로 오타, frame 0 이미지 미초기화 및 카메라 전환 이미지 cache 오류를 수정해야 합니다.",
        "검출 TXT와 ReID NPY는 동일한 행순서로 생성해야 합니다. GT를 검출이나 임베딩으로 대체하지 않습니다.",
        "가중치가 준비되지 않았습니다. ReID FeatureExtractor는 없는 checkpoint에 대해 pretrained 다운로드를 시도할 수 있으므로 실행 전 파일 존재를 검증해야 합니다.",
        "추적 FPS 30 고정, c005 ROI 제거, 카메라별 threshold, 수천 프레임의 anchor 선택 및 S001 특별 ID 처리 때문에 짧은 SCOUT clip으로 원본 전체 파이프라인을 실행할 수 없습니다.",
        "HOMO는 world→image 3×3 행렬입니다. SCOUT 원본 calibration만으로 자동 생성하지 않습니다. 렌즈 왜곡, 평면, 좌표계, 단위, 영상 크기를 검증해야 합니다.",
        "MMPose용 영상은 export frame 순서·크기와 일치해야 합니다. profile은 video.mp4 경로만 선언하며 영상 생성/원본 복사/리샘플링을 수행하지 않습니다.",
        "포즈 JSON에는 track ID가 없어 같은 frame의 SCT 행순서를 보존해야 합니다. 발 좌표 변환기와 STCRA 파일명·출력 경로 하드코딩 수정이 필요합니다.",
        "최종 제출 스크립트는 특정 scene 목록과 43개 카메라를 강제합니다. 공통 결과 변환기를 사용하고 camera-local SCT ID를 MTMC ID로 취급하지 않아야 합니다.",
    ]
    seen_sources = set()
    seen_model_ids = set()
    manifest_cameras = manifest.get("cameras", [])
    if not manifest_cameras:
        raise ValueError("UWIPL profile requires at least one camera")
    for camera in manifest_cameras:
        source_camera = camera["id"]
        if not isinstance(source_camera, str) or not source_camera:
            raise ValueError("UWIPL source camera id must be a nonempty string")
        model_id = _model_id(camera["model_camera_id"])
        if source_camera in seen_sources or model_id in seen_model_ids:
            raise ValueError("UWIPL source camera ids and model camera ids must be unique")
        seen_sources.add(source_camera)
        seen_model_ids.add(model_id)
        model_name = "c{:03d}".format(model_id)
        camera_root = "data/test/{}/{}".format(scene_name, model_name)
        frame_dirs = [camera_root + "/img", camera_root + "/frame"]
        directories.update(frame_dirs)
        cameras.append({
            "source_camera": source_camera,
            "model_camera_id": model_id,
            "model_camera_name": model_name,
            "frame_dirs": frame_dirs,
            "frame_pattern": "{frame:05d}.jpg",
            "video_path": camera_root + "/video.mp4",
            "calibration_path": None,
        })
        if camera.get("calibration") is None:
            blockers.append("{}: calibration이 없어 STCRA 공간 일관성 재할당을 구성할 수 없습니다.".format(source_camera))
        if (camera.get("width"), camera.get("height")) != (1920, 1080):
            blockers.append("{}: 영상 크기 {}×{}는 원본 검출·경계 제거의 1920×1080 가정과 다릅니다.".format(
                source_camera, camera.get("width"), camera.get("height")))
    if seen_model_ids != set(range(1, 8)):
        blockers.append("선택 카메라는 원본 S001의 c001..c007 구성과 다릅니다. 7개 카메라용 배열·보정·특별 처리 코드를 일반화해야 합니다.")
    sync = manifest.get("sync") or {}
    skew = sync.get("max_skew_seconds")
    tolerance = sync.get("tolerance_seconds", 0.05)
    if skew is None:
        blockers.append("카메라 간 timestamp skew가 확인되지 않았습니다. STCRA의 같은 frame=같은 시각 가정을 검증해야 합니다.")
    elif not isinstance(skew, (int, float)) or isinstance(skew, bool) or not math.isfinite(skew) or skew < 0:
        raise ValueError("max_skew_seconds must be a finite nonnegative number or null")
    elif not isinstance(tolerance, (int, float)) or isinstance(tolerance, bool) or not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance_seconds must be a finite nonnegative number")
    elif skew > tolerance:
        blockers.append("카메라 간 최대 skew {:.6f}s가 허용값 {:.6f}s를 초과합니다. 시간 정렬 없이 STCRA를 실행하면 안 됩니다.".format(skew, tolerance))
    notes = [
        "모델 frame은 0부터 시작하며 원본 source_frame/timestamp는 부모 manifest의 frame map으로 역변환합니다.",
        "img와 frame은 같은 선택 프레임을 가리킵니다. GT 기반 detection·embedding·예측 결과를 생성하지 않습니다.",
        "최상위 run_tracking.py는 project-root/data를 붙이지만 aic_track.py는 data-root를 직접 받습니다. ReID는 project-root, synthetic 검출은 절대 입력 경로를 사용합니다.",
        "선택 SCOUT sequence는 {}이며 모델 scene 별칭은 S001입니다. sequence 간 ID namespace는 공통 변환기에서 분리합니다.".format(manifest.get("sequence")),
        "SCOUT image_space={}를 유지합니다. 왜곡 보정이나 bbox/calibration 변환을 이 프로파일에서 수행하지 않습니다.".format(manifest.get("image_space", "raw")),
        "단일카메라 MOT 파일은 --camera와 local/global scope를 부모 CLI에서 명시해야 합니다. 파일명에서 camera나 scope를 추정하지 않습니다.",
    ]
    native_results = {
        "formats": ["aic9", "mot10", "mot16"],
        "frame_base": FRAME_BASE,
        "detections": {
            "path": "data/test_det/S001.txt",
            "columns": ["camera", "frame", "class", "x1", "y1", "x2", "y2", "score"],
            "class_person": 1,
            "generated": False,
        },
        "embeddings": {
            "directory": "data/test_emb",
            "s001_names": ["S001_{}.npy".format(name) for name in (
                "osnet_x1_0", "osnet_ibn_x1_0", "osnet_ain_x1_0",
                "osnet_x1_0_market", "osnet_x1_0_msmt17")],
            "alignment": "one feature vector per detection row, unchanged row order",
            "generated": False,
        },
        "sct": {"pattern": "data/SCT/S001_{camera}.txt", "format": "mot10", "scope": "camera_local"},
        "mtmc": {"pattern": "data/hungarian_cluster_S001/S001_{camera}.txt", "format": "mot10", "scope": "scene_global"},
        "pose": {"format": "json", "keypoints": "COCO17", "association": "frame and SCT row order; no track_id in upstream pose JSON"},
        "feet": {"format": "mot16", "extra_columns": ["left_ankle_x", "left_ankle_y", "left_ankle_score", "right_ankle_x", "right_ankle_y", "right_ankle_score"]},
        "submission": {"path": "data/submission.txt", "format": "aic9", "scope": "scene_global", "track_id": "scene_numeric * 1000 + scene_global_id; preserve verbatim"},
    }
    config = {
        "model": "uwipl", "source_commit": SOURCE_COMMIT,
        "source_sequence": manifest.get("sequence"),
        "scene_name": scene_name, "frame_base": FRAME_BASE,
        "image_space": manifest.get("image_space", "raw"),
        "inference_ready": False,
        "cameras": cameras, "native_results": native_results,
        "calibration": {"generated": False, "required_direction": "world_to_image", "world_units": None},
        "blockers": blockers, "notes": notes,
    }
    return {
        "name": "uwipl", "scene_name": scene_name, "frame_base": FRAME_BASE,
        "inference_ready": False, "cameras": cameras,
        "directories": sorted(directories),
        "config_files": {"configs/uwipl_adapter.json": config},
        "blockers": blockers, "notes": notes, "native_results": native_results,
    }


def _integer(text, field):
    try:
        value = Decimal(text.strip())
    except (InvalidOperation, ValueError) as error:
        raise ValueError("{} must be an integer".format(field)) from error
    if not value.is_finite() or value != value.to_integral_value():
        raise ValueError("{} must be a finite integer".format(field))
    return int(value)


def _number(text, field):
    try:
        value = float(text.strip())
    except ValueError as error:
        raise ValueError("{} must be numeric".format(field)) from error
    if not math.isfinite(value):
        raise ValueError("{} must be finite".format(field))
    return value


def _confidence(text, field, allow_missing=False):
    value = _number(text, field)
    if allow_missing and value == -1:
        return None
    if not 0 <= value <= 1:
        raise ValueError("{} must be in [0, 1]{}".format(field, " or -1" if allow_missing else ""))
    return value


def read_results(path: Path, *, format: str):
    """Yield native zero-based records; the parent maps frames, cameras and IDs.

    MOT camera is intentionally None. AIC scene-offset IDs are preserved.
    No ground truth, filename guessing, normalization, clipping or ID remapping.
    """
    if format not in FORMATS:
        raise ValueError("Unsupported UWIPL result format: {!r}".format(format))
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, strict=True)
        try:
            for fields in reader:
                try:
                    if len(fields) != FORMATS[format]:
                        raise ValueError("{} requires exactly {} columns; got {}".format(format, FORMATS[format], len(fields)))
                    if any(not field.strip() for field in fields):
                        raise ValueError("empty fields are not allowed")
                    if format == "aic9":
                        camera = _integer(fields[0], "camera_id")
                        track = _integer(fields[1], "track_id")
                        frame = _integer(fields[2], "frame")
                        bbox = [_number(fields[i], "bbox") for i in range(3, 7)]
                        if camera < 1:
                            raise ValueError("UWIPL camera_id must be positive")
                        if [_number(fields[i], "world_xy placeholder") for i in (7, 8)] != [-1.0, -1.0]:
                            raise ValueError("UWIPL aic9 expects -1,-1 placeholders; 2D world coordinates cannot be represented as world_xyz")
                        score, world = None, None
                    else:
                        camera = None
                        frame = _integer(fields[0], "frame")
                        track = _integer(fields[1], "track_id")
                        bbox = [_number(fields[i], "bbox") for i in range(2, 6)]
                        score = _confidence(fields[6], "score", allow_missing=True)
                        world = [_number(fields[i], "world_xyz") for i in range(7, 10)]
                        if world == [-1.0, -1.0, -1.0]:
                            world = None
                        if format == "mot16":
                            for i in (10, 11, 13, 14):
                                _number(fields[i], "ankle coordinate")
                            for i in (12, 15):
                                _confidence(fields[i], "ankle confidence")
                    if frame < 0 or track < 0:
                        raise ValueError("frame and track_id must be nonnegative")
                    if bbox[2] <= 0 or bbox[3] <= 0:
                        raise ValueError("bbox width and height must be positive")
                    yield {"camera_id": camera, "track_id": track, "frame": frame,
                           "bbox_xywh": bbox, "score": score, "world_xyz": world}
                except ValueError as error:
                    raise ValueError("{}:{}: {}".format(path, reader.line_num, error)) from error
        except csv.Error as error:
            raise ValueError("{}:{}: malformed CSV: {}".format(path, reader.line_num, error)) from error
