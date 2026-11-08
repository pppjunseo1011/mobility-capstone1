"""Compare available SCOUT metadata with pinned model input contracts.

This is a data assessment, not a model runner or a synchronization algorithm.
It neither invents calibration/labels nor decodes the full image dataset.
"""
import math
import statistics

from . import __version__
from .common import finite, new_output, within, write_json
from .dataset import TOOLKIT_COMMIT, metadata, scan_frames
from .profiles import PROFILES


GENERATIONS = {
    "implemented": "변환 기능 구현됨",
    "conditional": "조건 검증·추가 구현 필요",
    "inference_or_external": "모델 추론·외부 준비 필요",
}

# These contracts describe the source pinned in sources.lock.json. A generated
# path/config is not evidence that the unmodified model consumes that output.
REQUIREMENTS = [
    dict(id="frames", information="이미지·입력 폴더",
         scout="공식 로더는 images/sequence_N/cam_N/image_F.jpg를 사용",
         yachiyo="Original/scene_NNN/camera_NNNN/Frame/*.jpg",
         uwipl="data/test/S001/cNNN/img 및 frame/*.jpg",
         posetrack="dataset/test/scene_NNN/camera_NNNN/video.mp4",
         generation="implemented", implementation="export / 모델별 profile",
         conditions="실제 JPEG 필요. PoseTrack의 frames/는 중간 배치이며 직접 영상 입력이 아님."),
    dict(id="video", information="MP4 영상",
         scout="배포 README는 사전 추출 프레임(10FPS, 1920×1080)으로 설명",
         yachiyo="포즈 단계는 video.mp4", uwipl="선택한 포즈 경로의 video.mp4",
         posetrack="검출·포즈·ReID 모두 video.mp4",
         generation="implemented", implementation="export --videos --fps (ffmpeg/ffprobe)",
         conditions="연속 프레임과 명시 FPS 필요. 영상 인코딩은 촬영 시각 동기화가 아님."),
    dict(id="identifiers", information="카메라·프레임 번호",
         scout="cam_N, 원본 frame ID, sequence",
         yachiyo="camera_(N+1), 선택 구간은 frame 1부터",
         uwipl="c(N+1), 선택 구간은 frame 0부터",
         posetrack="camera_(N+1), 선택 영상 frame 0부터",
         generation="implemented", implementation="manifest.json / frame_map.csv / normalize",
         conditions="YACHIYO 포즈 내부의 1/0기준 차이는 추가 bridge 필요. 카메라 ID는 인물 ID와 다름."),
    dict(id="timestamps", information="촬영 시각·카메라 대응",
         scout="sequence 1은 장치명, sequence 2는 camN 키; 중복·누락 가능",
         yachiyo="원본 추적은 frame 번호 기반",
         uwipl="원본 코드의 FPS·frame/경로 가정 확인 필요",
         posetrack="동일 frame을 동시점으로 처리",
         generation="implemented", implementation="공통 metadata 로더 / assess / 원본 timestamp 보존",
         conditions="이름 대응·시간 차이 검사까지 구현. 실제 동기화는 다음 항목에서 별도 판단."),
    dict(id="synchronization", information="공통 시간축",
         scout="같은 frame 번호의 촬영 시각이 다를 수 있음",
         yachiyo="기본 fps=30 및 frame 단위 임계값",
         uwipl="frame_rate=30 및 장면별 규칙",
         posetrack="frame 단위 수명·다중 카메라 결합",
         generation="conditional", implementation="미구현: 시간 보간·리샘플링·새 대응표",
         conditions="구간/FPS/허용 오차/누락 정책 결정 후 timestamp로 정렬. GT와 결과도 같은 대응표 사용."),
    dict(id="projection", information="카메라 투영행렬 P",
         scout="JSON의 K(3×3), R(3×3), T(3), dist, view_id",
         yachiyo="camera projection matrix 등 모델 보정 형식",
         uwipl="카메라 기하를 원본 cam_utils 등의 규칙에 연결해야 함",
         posetrack="camera projection matrix (3×4)",
         generation="conditional", implementation="P=K[R|T] 참고값 계산은 구현; 모델용 파일은 미생성",
         conditions="P는 왜곡 없는 pinhole 기준. raw 영상에 직접 적용하면 안 되며 단위·좌표계 확인 필요."),
    dict(id="ground_geometry", information="지면 H·3D 장면 정보",
         scout="배포 README에 3D mesh/ground-only mesh 제공 명시; 실제 mesh는 아직 미검증",
         yachiyo="world→image H와 역투영",
         uwipl="cam_utils.HOMO 및 S001 카메라 기하가 하드코딩됨",
         posetrack="H 및 H[:,2] + 0.15*P[:,2] 발 평면",
         generation="conditional", implementation="H·mesh 변환·지면 적합 미구현",
         conditions="평면·세계 단위·왜곡 확인 후 H 생성 가능. 비평면이면 mesh 방식 검토와 모델 변경 필요. P/H 상대 스케일 보존."),
    dict(id="distortion", information="렌즈 왜곡·픽셀 좌표",
         scout="확인한 보정에는 0이 아닌 dist 존재",
         yachiyo="H 기반 역투영과 일치하는 픽셀 필요",
         uwipl="ROI/기하 및 특징 추출에 쓰는 픽셀 좌표 일치 필요",
         posetrack="왜곡 없는 P/H와 일치하는 픽셀 필요",
         generation="conditional", implementation="현재 raw 이미지 복사; undistortion·좌표 역변환 미구현",
         conditions="영상·검출·포즈·GT·결과 좌표계를 일관되게 처리. 이미지 보정만 추가하면 결과 역매핑이 불완전함."),
    dict(id="detections", information="검출 박스·confidence",
         scout="GT bbox는 평가용 정답이며 모델 검출값이 아님",
         yachiyo="YOLOX 검출 CSV", uwipl="YOLOX/MMYOLO 검출 및 행 순서",
         posetrack="frame,class,xyxy,score 7열 검출 파일",
         generation="inference_or_external", implementation="검출기는 환경 검증만 수행; 실제 추론 미실행",
         conditions="검출 가중치와 입력을 연결해 생성. GT를 검출 결과로 자동 대체하지 않음."),
    dict(id="pose_reid", information="포즈·ReID 특징",
         scout="모델 고유 포즈/embedding 파일로 제공된 것은 아님",
         yachiyo="검출과 일치하는 포즈 JSON·OSNet 특징",
         uwipl="여러 ReID 특징 및 SCT/포즈 결과의 행·ID 정렬",
         posetrack="57열 포즈 및 검출 행에 대응하는 ReID NPY",
         generation="inference_or_external", implementation="각 모델의 단계별 추론이 생성해야 함",
         conditions="모델별 단계 순서와 행·ID 대응 유지. UWIPL은 검출→ReID→SCT·clustering→포즈→STCRA. 특징을 임의 생성하거나 GT에서 추출하지 않음."),
    dict(id="scene_priors", information="ROI·카메라 연결·장면 규칙",
         scout="SCOUT 전용 설정을 실제 장면에서 검토해야 함",
         yachiyo="장면별 카메라 목록·FPS·추적 임계값",
         uwipl="S001 7카메라·ROI·인물 anchor·클러스터 규칙",
         posetrack="장면 선택·영상 경로·수명·해상도 가정",
         generation="conditional", implementation="카메라 목록만 기계적 생성; 나머지는 모델 연결 작업",
         conditions="AI City 수치를 재사용하지 않음. 평가 GT를 이용한 설정 누출 없이 공통 조건을 정해야 함."),
    dict(id="weights", information="학습 가중치",
         scout="데이터셋은 세 모델의 학습 가중치를 대신 제공하지 않음",
         yachiyo="YOLOX / OSNet / HRNet 등",
         uwipl="검출 / 여러 ReID / 포즈 가중치",
         posetrack="YOLOX / HRNet / aic24.pkl 등",
         generation="inference_or_external", implementation="자동 다운로드·가중치 로딩 미실행",
         conditions="팀원이 사용할 체크포인트·설정·해시를 확정하고 읽기 전용으로 연결."),
    dict(id="evaluation", information="정답·공통 추적 결과",
         scout="COCO/MOT bbox·사람 ID·world 정보; 라벨 품질/범위는 별도 확인",
         yachiyo="yachiyo-json / aic9 결과 파서",
         uwipl="mot10 / mot16 / aic9 결과 파서",
         posetrack="posetrack10 / aic9 결과 파서",
         generation="implemented", implementation="normalize / ground-truth → 분리된 CSV",
         conditions="변환 기능만 구현. 실제 결과·GT 범위 검증 및 평가 지표 계산은 미완료."),
]


def timing_summary(timestamps, cameras, tolerance):
    """Compare same frame numbers, explicitly retaining incomplete coverage."""
    tolerance = finite(tolerance, "timestamp tolerance")
    if tolerance < 0:
        raise ValueError("Timestamp tolerance must be >= 0")
    available = [row for row in cameras if row["timestamps_available"]]
    unavailable = [row["id"] for row in cameras if not row["timestamps_available"]]
    skews, incomplete = [], 0
    for frame in sorted(timestamps, key=int):
        row = timestamps[frame]
        values = [finite(row[c["timestamp_key"]], "timestamp") for c in available
                  if c["timestamp_key"] in row]
        if len(values) != len(available) or not available:
            incomplete += 1
        elif len(values) >= 2:
            skews.append(max(values)-min(values))
    ordered = sorted(skews)
    coverage_complete = len(available) >= 2 and not unavailable and incomplete == 0
    return {"policy": "same_source_frame_diagnostic_only; no resampling",
            "compared_cameras": [row["id"] for row in available],
            "unavailable_cameras": unavailable,
            "total_timestamp_frames": len(timestamps),
            "frames_with_all_compared_timestamps": len(skews),
            "frames_missing_compared_timestamps": incomplete,
            "cross_camera_comparison_possible": len(available) >= 2,
            "selected_camera_coverage_complete": coverage_complete,
            "max_skew_seconds": max(skews) if skews else None,
            "median_skew_seconds": statistics.median(skews) if skews else None,
            "p95_skew_seconds": ordered[max(0, math.ceil(.95*len(ordered))-1)] if ordered else None,
            "tolerance_seconds": tolerance,
            "frames_over_tolerance": sum(value > tolerance for value in skews),
            "all_selected_same_frame_within_tolerance": (
                max(skews) <= tolerance if coverage_complete and skews else None),
            "synchronization_performed": False}


def assessment_markdown(report):
    data, timing = report["data"], report["timing"]
    lines = ["# SCOUT → 세 모델 입력 요구사항 비교", "",
             "이 보고서는 지정 데이터 경로에서 메타데이터를 검사한 결과다. 이미지 디코딩·모델 추론·평가를 수행하지 않는다.", "",
             "## 실제 입력에서 확인한 범위", "",
             "- sequence: `{}`".format(report["sequence"]),
             "- 선택 보정 카메라: {}개; timestamp 대응 가능: {}개".format(data["camera_count"], data["timestamp_camera_count"]),
             "- timestamp 미확인 카메라: {}".format(", ".join(timing["unavailable_cameras"]) or "없음"),
             "- JPEG 파일이 있는 카메라: {}개 (픽셀 디코딩 미검증)".format(data["cameras_with_images"]),
             "- 비영 왜곡 계수 카메라: {}개".format(data["nonzero_distortion_cameras"]),
             "- timestamp frame 키: {}개".format(timing["total_timestamp_frames"]),
             "- 비교된 카메라: {}".format(", ".join(timing["compared_cameras"]) or "없음"),
             "- 같은 frame의 최대 촬영 시각 차이: {}초".format(timing["max_skew_seconds"]),
             "- 허용값 {}초 초과: {} / {} frame".format(timing["tolerance_seconds"], timing["frames_over_tolerance"], timing["frames_with_all_compared_timestamps"]),
             "- 위 시각 차이는 비교 가능한 카메라의 timestamp가 모두 있는 frame만 집계한다. 누락 카메라를 0초 오차로 처리하지 않는다.",
             "- 허용값은 검사 설정값이며 확정된 연구 평가 기준이 아니다.", "",
             "## 제공 정보와 모델 요구사항", "",
             "| 정보 | SCOUT 제공/확인 | YACHIYO | UWIPL | PoseTrack | 생성·적용 분류 |",
             "|---|---|---|---|---|---|"]
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    for item in report["requirements"]:
        lines.append("| " + " | ".join(cell(item[key]) for key in (
            "information", "scout", "yachiyo", "uwipl", "posetrack")) + " | " + GENERATIONS[item["generation"]] + " |")
    lines.extend(["", "## 추가 생성·적용 조건", ""])
    for item in report["requirements"]:
        lines.extend(["- **{}**: {}. {}".format(item["information"], item["implementation"], item["conditions"])])
    lines.extend(["", "## 실제 추론 전 남은 항목", ""])
    lines.extend("- " + value for value in report["blockers"])
    lines.extend(["", "## 기준 및 한계", "",
                  "- SCOUT toolkit 기준 커밋: `{}`".format(TOOLKIT_COMMIT),
                  "- 모델 소스: sources.lock.json 및 docs/adapters/{yachiyo,uwipl,posetrack}.md의 코드 근거.",
                  "- 구현 상태와 현재 데이터의 존재 여부는 별개다. MP4 생성 기능이 있어도 실제 JPEG를 확보해야 실행할 수 있다.",
                  "- model_inference_ready=false; 실제 라벨 품질, 전체 이미지, mesh, world 단위·평면은 미검증.", ""])
    return "\n".join(lines)


def assess_dataset(root, sequence=1, cameras=None, overrides=None, tolerance=.05, output=None):
    root, timestamps, rows = metadata(root, sequence, cameras, overrides, allow_unavailable=True)
    if not rows:
        raise ValueError("No calibration cameras found")
    timing = timing_summary(timestamps, rows, tolerance)
    for row in rows:
        folder = within(root, "images/sequence_{}/{}".format(sequence, row["id"]))
        frames = scan_frames(folder) if folder.is_dir() else {}
        row.update(images_present=bool(frames), image_count=len(frames),
                   image_frame_range=[min(frames), max(frames)] if frames else None,
                   image_pixels_decoded=False)
        row.pop("calibration")
    blockers = ["실제 가중치·모델 실행기 연결과 AI City 전용 규칙 수정이 필요함.",
                "왜곡·world 단위·지면/mesh 검증 전에는 모델용 P/H를 적용할 수 없음.",
                "동기화 정책과 GT 적용 범위·품질을 확정하지 않음."]
    if any(not row["images_present"] for row in rows):
        blockers.append("선택 카메라 중 실제 JPEG가 없는 경로가 있음; 현재 입력만으로 export할 수 없음.")
    if timing["unavailable_cameras"] or timing["frames_missing_compared_timestamps"]:
        blockers.append("선택 카메라의 timestamp 대응 또는 일부 시각이 누락됨; 대상 선택/보완 필요.")
    if timing["frames_over_tolerance"]:
        blockers.append("같은 frame 번호의 카메라 시각 차이가 검사 허용값을 초과함; 그대로 동시점 처리 불가.")
    if any(row["backward_timestamp_steps"] for row in rows):
        blockers.append("시간 역행이 존재함; 원본 및 시간 대응을 확인해야 함.")
    report = {"schema_version": 1, "adapter_version": __version__, "source_root": str(root),
              "sequence": "sequence_{}".format(sequence), "toolkit_commit": TOOLKIT_COMMIT,
              "scope": "metadata_and_input_contract_assessment",
              "data": {"camera_count": len(rows),
                       "timestamp_camera_count": sum(row["timestamps_available"] for row in rows),
                       "cameras_with_images": sum(row["images_present"] for row in rows),
                       "nonzero_distortion_cameras": sum(row["geometry"]["distortion_nonzero"] for row in rows),
                       "coco_file_present": within(root, "annotations/coco_format/annotations_coco_sequence_{}.json".format(sequence)).is_file(),
                       "mot_files_present": [row["id"] for row in rows if within(root, "annotations/mot_format/sequence_{}/{}.txt".format(sequence, row["id"])).is_file()],
                       "annotation_content_validated": False, "mesh_validated": False},
              "cameras": rows, "timing": timing,
              "model_frame_bases": {name: profile.FRAME_BASE for name, profile in PROFILES.items()},
              "requirements": REQUIREMENTS, "blockers": blockers,
              "model_inference_ready": False, "model_inference_run": False,
              "full_dataset_validated": False, "ground_truth_used_as_input": False}
    if output:
        with new_output(output, root) as staging:
            write_json(staging / "compatibility.json", report)
            (staging / "comparison.md").write_text(assessment_markdown(report), encoding="utf-8")
    return report
