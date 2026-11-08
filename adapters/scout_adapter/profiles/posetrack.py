"""PoseTrack master ba727e6 input layout and native tracking result reader.

This module describes staging paths; it neither creates videos nor infers camera
geometry. Native PoseTrack does not read the staged JPEG directory or this
module's adapter metadata directly.
"""

import math
from decimal import Decimal, InvalidOperation
from pathlib import Path


SOURCE_COMMIT = "ba727e6bc6e45e44cd6e006b73e00864892173c2"
FRAME_BASE = 0
_COLUMNS = [
    "camera_id", "track_id", "frame", "x", "y", "width", "height",
    "world_x", "world_y", "position_source",
]


def _manifest_integer(value, field, minimum=0, maximum=None):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError("{} must be an integer >= {}".format(field, minimum))
    if maximum is not None and value > maximum:
        raise ValueError("{} must be <= {}".format(field, maximum))
    return value


def build_profile(manifest):
    """Return a relative, non-executing staging plan for one SCOUT sequence."""
    number = _manifest_integer(manifest.get("sequence_number"), "sequence_number")
    # Match detection/get_detection.py, including its literal leading zero.
    scene_name = "scene_0{:02d}".format(number)
    scene_dir = "dataset/test/{}".format(scene_name)
    cameras = []
    model_ids = set()
    source_ids = set()
    directories = [
        scene_dir,
        "result/detection/{}".format(scene_name),
        "result/pose/{}".format(scene_name),
        "result/reid/{}".format(scene_name),
        "result/track",
        "result/track_log",
        "config",
    ]
    blockers = [
        "각 camera_XXXX/video.mp4 생성과 실제 디코딩·프레임 역매핑 검증이 필요하다. "
        "frames/의 JPEG는 어댑터 중간 산출물이며 원본 모델이 직접 읽지 않는다.",
        "calibration.json의 world→image P(3×4), ground→image H(3×3)는 자동 생성하지 않는다. "
        "SCOUT 좌표 단위·축·지면 z=0, 왜곡 처리, 발 높이 0.15 가정을 검증해야 한다.",
        "내보낸 dataset/test 및 result를 저장소 기준 경로에 연결하는 실행 workspace와 "
        "검출·포즈·ReID 가중치가 필요하다. 실제 모델 추론은 검증하지 않았다.",
    ]
    source_cameras = manifest.get("cameras")
    if not isinstance(source_cameras, list) or not source_cameras:
        raise ValueError("manifest.cameras must be a nonempty list")
    for camera in source_cameras:
        source_id = camera.get("id")
        if not isinstance(source_id, str) or not source_id:
            raise ValueError("camera.id must be a nonempty string")
        model_id = _manifest_integer(
            camera.get("model_camera_id"), "model_camera_id", 1, 9999
        )
        if source_id in source_ids or model_id in model_ids:
            raise ValueError("source camera and model camera IDs must be unique")
        source_ids.add(source_id)
        model_ids.add(model_id)
        model_name = "camera_{:04d}".format(model_id)
        camera_dir = "{}/{}".format(scene_dir, model_name)
        frame_dir = "{}/frames".format(camera_dir)
        cameras.append({
            "source_camera": source_id,
            "model_camera_id": model_id,
            "model_camera_name": model_name,
            "frame_dirs": [frame_dir],
            "frame_pattern": "{frame:06d}.jpg",
            "video_path": "{}/video.mp4".format(camera_dir),
            "calibration_path": "{}/calibration.json".format(camera_dir),
        })
        directories.extend([camera_dir, frame_dir])
        if (camera.get("width"), camera.get("height")) != (1920, 1080):
            blockers.append(
                "{}: 해상도가 1920×1080으로 확인되지 않았다. 검출의 고정 배율, "
                "ReID 클리핑, 추적의 y=1078 경계를 수정·검증해야 한다.".format(source_id)
            )
        if camera.get("calibration") is None:
            blockers.append("{}: 원본 카메라 보정 정보가 없다.".format(source_id))

    image_space = manifest.get("image_space", "raw")
    if image_space != "undistorted":
        blockers.append(
            "입력 image_space={!r}: 원본 모델의 P/H 기하는 렌즈 왜곡을 보정하지 않는다. "
            "이미지와 좌표·보정행렬의 일관된 왜곡 처리가 필요하다.".format(image_space)
        )
    sync = manifest.get("sync") or {}
    skew = sync.get("max_skew_seconds")
    tolerance = sync.get("tolerance_seconds", 0.05)
    if skew is None:
        blockers.append("카메라 간 timestamp 동기화 오차가 확인되지 않았다.")
    elif not isinstance(skew, (int, float)) or not math.isfinite(skew) or skew < 0:
        raise ValueError("sync.max_skew_seconds must be finite and nonnegative")
    elif (not isinstance(tolerance, (int, float)) or
          not math.isfinite(tolerance) or tolerance < 0):
        raise ValueError("sync.tolerance_seconds must be finite and nonnegative")
    elif skew > tolerance:
        blockers.append(
            "카메라 간 timestamp 최대 오차 {}초가 허용값 {}초를 넘는다. "
            "원본 추적기는 timestamp를 읽지 않고 같은 frame 번호를 동시점으로 본다."
            .format(skew, tolerance)
        )

    notes = [
        "모든 경로는 export 출력 디렉터리 기준 상대 경로다. 원본 저장소를 수정하지 않는다.",
        "frame은 선택한 source_frame 목록의 0-based 순번이다. 원본 번호와 timestamp의 "
        "역매핑은 공통 manifest를 사용하며 영상 생성 시 순서를 그대로 유지한다.",
        "검출은 쉼표 7열 frame,class=1,x1,y1,x2,y2,score; 포즈는 공백 57열 "
        "frame,x1,y1,x2,y2,score,17×(x,y,score); ReID는 검출 행 순서의 N×2048 .npy다.",
        "calibration·검출·포즈·ReID의 카메라 집합과 정렬 순서 및 검출별 행 순서가 같아야 한다. "
        "빈 파일·단일행 np.loadtxt 처리도 원본 실행 코드에서 보완해야 한다.",
        "Pose/ReID --start/--end는 정렬된 scene 목록의 Python slice다. 단일 scene은 "
        "--start 0 --end 1이며 기본 --end -1은 마지막 scene을 제외한다.",
        "원본 shell의 GPU/CPU 배정과 scene 범위를 그대로 실행하지 않는다. "
        "추적 임계값은 프레임 수 기준이므로 SCOUT의 실제 FPS에 대한 영향도 검토한다.",
        "track_id는 scene마다 1부터 시작한다. 공통 ID 범위는 sequence와 함께 해석한다.",
        "config/posetrack_adapter.json은 어댑터 설명용이며 원본 모델이 읽는 설정 파일이 아니다.",
    ]
    native_results = {
        "default_format": "posetrack10",
        "frame_base": FRAME_BASE,
        "track_id_scope": "sequence",
        "formats": {
            "posetrack10": {
                "path": "result/track/{}.txt".format(scene_name),
                "delimiter": "whitespace",
                "columns": list(_COLUMNS),
            },
            "aic9": {
                "path": "result/track.txt",
                "delimiter": "whitespace",
                "columns": list(_COLUMNS[:-1]),
            },
        },
        "score": None,
        "world_coordinates": "world_xy only; no z is provided",
        "position_source": {"1": "bbox ground point", "2": "ankle plane", "3": "3D joints XY"},
    }
    metadata = {
        "kind": "adapter_metadata_not_upstream_config",
        "source_commit": SOURCE_COMMIT,
        "source_sequence": manifest.get("sequence"),
        "scene_name": scene_name,
        "frame_base": FRAME_BASE,
        "image_space": image_space,
        "video_required": True,
        "calibration_generated": False,
        "ready": False,
        "cameras": cameras,
        "native_results": native_results,
        "blockers": blockers,
    }
    return {
        "name": "posetrack",
        "scene_name": scene_name,
        "frame_base": FRAME_BASE,
        "ready": False,
        "cameras": cameras,
        "directories": directories,
        "config_files": {"config/posetrack_adapter.json": metadata},
        "blockers": blockers,
        "notes": notes,
        "native_results": native_results,
    }


def _result_integer(token, field, minimum, maximum=None):
    try:
        value = Decimal(token)
    except InvalidOperation:
        raise ValueError("{} must be numeric".format(field)) from None
    if not value.is_finite() or value != value.to_integral_value():
        raise ValueError("{} must be a finite integer".format(field))
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError("{} is outside the supported range".format(field))
    return int(value)


def _result_float(token, field):
    try:
        value = float(token)
    except ValueError:
        raise ValueError("{} must be numeric".format(field)) from None
    if not math.isfinite(value):
        raise ValueError("{} must be finite".format(field))
    return value


def read_results(path: Path, *, format: str):
    """Read strict native whitespace rows, retaining frame-zero and local IDs.

    ``posetrack10`` is the per-scene tracking file; ``aic9`` is the submission
    file. Neither contains confidence or a 3D Z coordinate. Blank lines and
    whole-line ``#`` comments are ignored; malformed data raises ValueError
    including its path and line number. Empty files yield no rows.
    """
    if format not in ("posetrack10", "aic9"):
        raise ValueError("unsupported PoseTrack result format: {!r}".format(format))
    path = Path(path)
    expected = 10 if format == "posetrack10" else 9
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            tokens = stripped.split()
            try:
                if len(tokens) != expected:
                    raise ValueError("expected {} whitespace-separated columns, got {}".format(
                        expected, len(tokens)
                    ))
                camera_id = _result_integer(tokens[0], "camera_id", 1, 9999)
                track_id = _result_integer(tokens[1], "track_id", 1)
                frame = _result_integer(tokens[2], "frame", 0)
                bbox = [_result_float(token, field) for token, field in zip(
                    tokens[3:7], _COLUMNS[3:7]
                )]
                if bbox[2] <= 0 or bbox[3] <= 0:
                    raise ValueError("bbox width and height must be positive")
                world_xy = [_result_float(token, field) for token, field in zip(
                    tokens[7:9], _COLUMNS[7:9]
                )]
                record = {
                    "camera_id": camera_id,
                    "track_id": track_id,
                    "frame": frame,
                    "bbox_xywh": bbox,
                    "score": None,
                    "world_xyz": None,
                    "world_xy": world_xy,
                }
                if format == "posetrack10":
                    record["position_source"] = _result_integer(
                        tokens[9], "position_source", 1, 3
                    )
            except ValueError as error:
                raise ValueError("{}:{}: {}".format(path, line_number, error)) from error
            yield record
