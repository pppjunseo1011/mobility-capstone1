#!/usr/bin/env python3
"""Generate the uncapped OSC handoff from audited results, never hardcoded metrics."""
import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/'outputs/osc-pilot-uncapped-20261101'
INF = ROOT/'outputs/osc-inference-uncapped-20261101'
DEST = ROOT/'docs/handoff'
STEM = 'OSC_SCOUT_인계서_제한해제_20261101'
DEST.mkdir(parents=True,exist_ok=True)

def load(name): return json.loads((RUN/name).read_text())
def number(n): return f'{n:,}'
def percent(n): return f'{n*100:.2f}%'
def table(head, rows, cls=''):
    return '<table class="'+cls+'"><thead><tr>'+''.join('<th>'+v+'</th>' for v in head)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(v)+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table>'
def pic(path): return 'data:image/jpeg;base64,'+base64.b64encode((RUN/path).read_bytes()).decode()
def page(n,title,body):
    return f'<section class="page"><div class="eyebrow">SCOUT × OSC · 인원 제한 해제 재검증</div><h1>{title}</h1>{body}<footer><span>2026.11.01 · 5초 개발용 파일럿 · 공식 벤치마크 결과 아님</span><b>{n} / 5</b></footer></section>'

scope=load('validation_scope.json');prep=load('prepare_report.json')
e=load('evaluation/evaluation_report.json');repeat=load('tracking/repeat_report.json')
c=load('comparison.json');old=c['baseline'];new=c['uncapped'];world=prep['world']
det=json.loads((INF/'yachiyo/detection/report.json').read_text())
assert c['status']=='PASS' and scope['max_people_per_frame'] is None and repeat['identical_fixed_whole']
N=number(new['selected']);L=number(new['local_rows']);G=number(new['global_rows'])
global_counts=' · '.join(f'{k}: {v}' for k,v in new['global_rows_per_camera'].items())
status=[
('01','SCOUT 프레임·GT','완료','대표 GT BBox 시각화 24장'),
('02','검증 구간','완료','sequence_1 / cam_0~7 / 원본 frame 277~326'),
('03','카메라·프레임 매핑','완료','camera_0001~8 / OSC frame 1~50'),
('04','검출·임베딩','완료',f'{N}×512 · 개수 제한 없음 · NaN/Inf 없음'),
('05','Pose 대응','완료',f'{N}개 BBox–Keypoint serial 대응 통과'),
('06','World·별도 GT','부분 완료','위치 오차 비교 완료 / 물리 단위·보정 적합성 미확정'),
('07','SCPT·Local ID','완료',f'{L}행 할당 · 보정 전후 결과 보존'),
('08','MCPT·Global ID','제한 실행',f'{G}행 · {e["global_ids"]} ID / 공간 거리 보정 OFF'),
('09','공통 CSV','완료','camera/frame/bbox/id 검사 · 중복 없음'),
('10','프로토콜별 평가','파일럿 완료','2D·multi-view 분리 / 공식 SCOUT·HOTA 미실시'),
('11','반복·기록','완료','추적 2회 · 최종 JSON 해시 동일'),
]
pages=[page(1,'OSC 담당자 전달 요약',f'''
<p class="lead"><b>프레임당 상위 10개 제한을 해제</b>하고 같은 400장을 검출 → Pose·ReID → SCPT → MCPT → 평가까지 다시 처리했습니다.</p>
<div class="stats"><div><strong>8대 · 400장</strong><span>카메라당 50프레임</span></div><div><strong>{N} × 512</strong><span>제한 해제 후 임베딩</span></div><div><strong>{G}행 · {e['global_ids']} ID</strong><span>최종 Global 출력</span></div></div>
<div class="callout"><b>이번에 바꾼 조건은 인원 수 상한뿐입니다.</b><br>objectness ≥0.3, 5초 구간, 가중치, 추적 설정을 유지했습니다. 이전 실행과 원시 검출·원본 이미지·GT·카메라 매핑이 같은지 대조했습니다.</div>
<h2>11개 체크리스트 확인 결과</h2>{table(['번호','항목','상태','이번 결과'],status,'checklist')}
<p class="small">SCPT: 카메라 내부 추적. MCPT: 카메라 사이 동일 인물 연결. GT: 제공된 정답 주석. 실행과 형식 검증의 성공이 충분한 추적 성능 확보를 의미하지는 않습니다.</p>
''')]

comparison_rows=[]
for label,key,fmt in [('후속 입력 검출 수','selected',number),('Local ID 할당 행','local_rows',number),
                     ('최종 Global ID 행','global_rows',number),('Global ID 수','global_ids',number),
                     ('Local 2D IDF1','local_idf1',percent),('Local 2D recall','local_recall',percent),
                     ('Local 2D MOTA','local_mota',percent),('Local ID 전환 횟수','local_id_switches',number),
                     ('Global 결과의 카메라별 2D IDF1','global_2d_idf1',percent),
                     ('Multi-view Global IDF1','multiview_idf1',percent)]:
    comparison_rows.append((label,fmt(old[key]),fmt(new[key])))
pages.append(page(2,'같은 구간에서 제한 전후 비교',f'''
<h2>01 · 검증 범위와 유지한 조건</h2>
{table(['항목','내용'],[('구간','sequence_1 / cam_0~cam_7 / source frame 277~326, 양끝 포함'),('번호 대응','cam_0→camera_0001 … cam_7→camera_0008 / OSC frame = source frame − 276'),('규모·정답',f'1920×1080 · 약 10 FPS · 400장 / GT {prep["gt_rows"]:,}행 · {prep["gt_ids"]} ID'),('고정 설정','objectness ≥0.3 / fps=10 / short_track_th=40 / 공간 보정·Global 재할당 OFF')])}
<p class="small">첫 8카메라 GT에는 보간이 포함됩니다. 같은 frame의 카메라 간 timestamp 차이는 최대 약 0.032초입니다. GT는 평가에만 사용했고 추적 컨테이너에는 연결하지 않았습니다.</p>
<h2>02 · 제한 해제 후 달라진 값</h2>{table(['항목','이전: 최대 10개','이번: 개수 상한 없음'],comparison_rows)}
<div class="note">원시 검출 <b>{det['detections']:,}개</b>는 이전과 정확히 동일했습니다. 신뢰도 기준을 통과한 <b>{N}개 전부</b>를 후속 입력으로 사용했습니다. 400장 중 <b>{c['frames_exceeding_ten']}장</b>에서 10개를 초과했고, 프레임별 실제 입력 수는 <b>{c['min_selected_per_frame']}~{c['max_selected_per_frame']}개</b>입니다.</div>
<p class="small">상한 해제는 모든 사람이 검출된다는 뜻이 아닙니다. 검출기의 신뢰도 기준·NMS는 유지했습니다. 비교표는 같은 5초 구간의 설정 비교이며 세 모델의 성능 순위를 나타내지 않습니다.</p>
'''))

pages.append(page(3,'결과 해석과 남은 검증',f'''
<h2>01 · 미할당과 카메라별 출력</h2>
{table(['확인 항목','이번 결과'],[('Local ID 미할당',f'{new["local_unassigned"]:,}행'),('Global 보정 단계에서 제거',f'{new["global_removed"]:,}행 — raw JSON과 제거 목록 보존'),('카메라별 Global 행',global_counts),('Local ID 수 (cam_0~7 순)', ' · '.join(map(str,new['local_ids_per_camera'])))])}
<p class="small">SCOUT <b>cam_5는 이번에도 최종 Global 출력이 없습니다.</b> 5초 구간에서 4초 대표 tracklet 조건을 유지하므로 짧거나 단절된 tracklet은 제외될 수 있습니다. 미할당 ID를 임의의 새 ID로 대체하지 않았습니다.</p>
<h2>02 · World 좌표 검증</h2>
<p>예측 BBox 하단을 왜곡 보정 후 ground mesh에 투영했습니다. 이후 별도 MOT GT의 XYZ와 비교했으며 GT 좌표를 예측에 주입하지 않았습니다.</p>
{table(['항목','결과'],[('좌표 생성 상태',' · '.join(f'{k}: {v:,}' for k,v in world['prediction_status_counts'].items())),('GT 대응 / World 비교',f'IoU≥0.5 BBox {world["bbox_matched"]:,}쌍 / 양쪽 좌표 유효 {world["world_compared"]:,}쌍'),('XYZ 오차 평균 / 중앙값 / p95',f'{world["mean_error_dataset_units"]:.3f} / {world["median_error_dataset_units"]:.3f} / {world["p95_error_dataset_units"]:.3f} 데이터셋 단위')])}
<p class="small">대응된 부분집합의 사람 위치 오차입니다. 카메라 보정만의 오차나 전체 성능이 아니며, 물리 단위를 m/cm로 확정하지 않았습니다.</p>
<h2>03 · 평가 범위와 반복 실행</h2>
<ul><li>2D는 각 camera/frame 안에서 IoU≥0.5로 평가합니다. Multi-view는 view별 대응 후 모든 카메라에 걸쳐 Global ID 일관성을 계산합니다.</li><li>World XY 진단은 별도 JSON에 기록했습니다. 공식 SCOUT 평가와 HOTA는 미실시입니다.</li><li>추적 2회 최종 JSON SHA-256이 동일합니다. SCPT+MCPT 시간은 <b>{repeat['runs'][0]['total_seconds']:.3f}초 / {repeat['runs'][1]['total_seconds']:.3f}초</b>입니다. 신경망 추론·준비·평가 시간을 포함한 전체 FPS가 아닙니다.</li></ul>
<div class="callout warning">이번 재검증은 <b>인원 제한을 해제한 5초 파일럿</b>입니다. 공간 조건을 포함한 원본 OSC 전체 설정 검증과 더 긴 구간에서의 성능 평가가 남아 있습니다.</div>
'''))

figs=[('GT BBox','제공된 정답 주석입니다. ID는 GT 번호입니다.','visualizations/gt/cam_0_277.jpg'),
      ('Local ID','제한 해제 후 카메라 내부 추적 결과입니다.','evaluation/visualizations/local_cam_0_277.jpg'),
      ('Global ID','카메라 사이 연결 및 보정 후 최종 결과입니다.','evaluation/visualizations/global_cam_0_277.jpg')]
fig_html=''.join(f'<div class="figure"><div class="figlabel"><h3>{t}</h3><p>{v}</p></div><img src="{pic(p)}" alt="{t}"></div>' for t,v,p in figs)
pages.append(page(4,'제한 해제 후 실제 시각화',f'''
<p class="lead">sequence_1 · cam_0 · 원본 frame 277</p><p class="small">같은 이미지의 GT / Local / Global 출력을 비교합니다. 세 이미지의 ID 숫자는 직접 대응하는 번호가 아닙니다. 원본 해상도 이미지를 삽입해 확대할 수 있습니다.</p>
{fig_html}<div class="note">첨부 시각화는 GT 24장 + Local 24장 + Global 24장 = 72장입니다. 8개 카메라의 source frame 277·301·326을 포함합니다.</div>
'''))

pages.append(page(5,'전달 자료와 후속 작업',f'''
<h2>01 · ZIP을 열면 START_HERE.txt부터 확인</h2>
<p>결과 검토용 묶음입니다. Docker 이미지·가중치·전체 원본 데이터는 제외했습니다. 아래 기준 경로는 <code>scout-mtmc/outputs/osc-pilot-uncapped-20261101/</code>입니다.</p>
{table(['용도','상대 경로'],[('범위·매핑','validation_scope.json<br>scout_to_osc_camera_map.json'),('검출·포즈·임베딩','inputs/native/ · inputs/Pose/ · inputs/features/'),('Local / Global 결과','tracking/repeat_01/fixed_camera…json<br>tracking/repeat_01/fixed_whole_tracking_results.json'),('공통 CSV','evaluation/local_predictions.csv<br>evaluation/global_common/predictions.csv'),('평가·좌표·이전 비교','evaluation/metrics_*.json · world_gt_comparison.json<br>comparison.json · comparison.csv · baseline_reference/'),('설정·시간·검증 근거','tracking/repeat_01/settings.json · tracking/repeat_report.json<br>provenance/evidence.json')],'files')}
<div class="note"><b>보정 전후를 구분하세요.</b> camera…json / whole…json은 raw 결과이며 <b>fixed_</b> 접두사가 최종 보정 결과입니다. raw와 repeat_02도 함께 보존했습니다. JSON의 /work 경로는 기존 컨테이너 경로입니다.</div>
<h2>02 · 코드와 기록</h2>
<p class="small">모델: riips/AIC24_Track1_YACHIYO_RIIPS<br>커밋: <code>f881fe0edf60fe697efb2bf4c935b80a80861230</code><br>가중치·원본 GT·보정·mesh·스크립트·출력의 해시, Docker 이미지 ID와 실제 실행 명령을 기록했습니다. 기존 10개 제한 결과는 비교 근거로 서버에 보존했습니다.</p>
<h2>03 · 다음에 결정할 사항</h2>
<ol><li>인원 제한 없는 더 긴 연속 구간으로 범위를 확장합니다.</li><li>Local ID 단절 및 Global ID 미할당 원인을 분석합니다.</li><li>World 단위·좌표 실패 처리·허용 오차를 확정합니다.</li><li>공간 조건을 포함한 MCPT와 독립 검증 구간을 준비합니다.</li><li>공식 SCOUT / HOTA 평가 조건을 확정한 뒤 모델을 비교합니다.</li></ol>
'''))

css='''
@page { size: A4; margin: 0; }
* { box-sizing: border-box; }
body { margin:0; color:#233046; background:#e8edf3; font-family:"Noto Sans CJK KR",sans-serif; font-size:10pt; line-height:1.55; }
.page { width:210mm; height:297mm; padding:15mm 18mm 19mm; margin:8mm auto; background:white; position:relative; break-after:page; }
.page:last-child { break-after:auto; }
.eyebrow { color:#246b80; letter-spacing:.06em; font-size:8.5pt; font-weight:700; margin-bottom:4mm; }
h1 { color:#142b46; font-size:23pt; line-height:1.25; margin:0 0 6mm; letter-spacing:-.04em; }
h2 { font-size:12.2pt; color:#185d75; border-bottom:1px solid #d6e3e8; padding-bottom:1.5mm; margin:5mm 0 2.5mm; }
h3 { font-size:11pt; margin:0 0 2mm; color:#185d75; }
p { margin:0 0 3mm; } .lead { font-size:11pt; line-height:1.65; }
.small { font-size:8.5pt; color:#566277; line-height:1.6; }
.stats { display:flex; gap:3mm; margin:5mm 0; }.stats div { flex:1; background:#edf5f7; border-radius:2mm; padding:3mm; }.stats strong {display:block;color:#164c64;font-size:16pt;}.stats span{font-size:8.5pt;color:#526477;}
.callout { background:#edf4fa; border-left:3px solid #206780; padding:3mm 4mm; margin:4mm 0; font-size:9.5pt; }
.warning { background:#fff7e7; border-color:#bd842b; }
.note { background:#f4f6f8; padding:3mm 4mm; font-size:9pt; margin:3mm 0; }
table { width:100%; border-collapse:collapse; font-size:9pt; margin:2mm 0 3mm; line-height:1.5; }
th { background:#193d56; color:white; text-align:left; padding:2.2mm 2.4mm; font-weight:500; }
td { padding:2.2mm 2.4mm; border-bottom:1px solid #dee5eb; vertical-align:top; }
tr:nth-child(even) td { background:#f6f8fa; }
.checklist { font-size:8.3pt; }.checklist td {padding:2.1mm 1.8mm;}.checklist th:nth-child(1){width:8%;}.checklist th:nth-child(2){width:25%;}.checklist th:nth-child(3){width:15%;}
.files {font-size:8.4pt;}.files td:first-child{width:28%;font-weight:500;}.files td:nth-child(2){font-family:"Noto Sans CJK KR",sans-serif;overflow-wrap:anywhere;}
ul,ol { margin:2mm 0 3mm; padding-left:5.5mm; } li { margin-bottom:2mm; }
code {font-family:"Noto Sans CJK KR",sans-serif;font-size:8.5pt;background:#f0f3f6;padding:0 1mm;overflow-wrap:anywhere;}
.figure { display:flex; gap:3mm; align-items:flex-start; margin:3mm 0; padding-bottom:2mm; border-bottom:1px solid #dde4eb; }
.figlabel {width:59mm;flex-shrink:0;}.figlabel p {font-size:8.5pt;color:#536176;}.figure img {width:110mm;height:auto;display:block;}
footer {position:absolute;left:18mm;right:18mm;bottom:9mm;border-top:1px solid #dce3eb;padding-top:2mm;display:flex;justify-content:space-between;font-size:7.7pt;color:#627287;}
@media print {body{background:white;}.page{margin:0;} }
'''

document='<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8"><title>SCOUT OSC 인원 제한 해제 재검증</title><style>'+css+'</style></head><body>'+''.join(pages)+'</body></html>'
path=DEST/(STEM+'.html')
path.write_text(document)
print(path)

md_rows='\n'.join('| '+' | '.join(map(str,row))+' |' for row in comparison_rows)
status_md='\n'.join('| '+' | '.join(row)+' |' for row in status)
details=f'''# OSC 인원 제한 해제 재검증 — 2026-11-01

프레임당 10개 상한을 해제하고 sequence_1 / cam_0~cam_7 / source frame 277~326의
400장을 검출·Pose·ReID부터 다시 처리했다. SCPT·MCPT는 2회 실행하고 평가·시각화를 갱신했다.
현재 결과: [outputs/osc-pilot-uncapped-20261101](../outputs/osc-pilot-uncapped-20261101).
이전 10개 제한 결과는 `outputs/osc-pilot-20261101`에 비교 근거로 보존한다.

## 인계 자료

- [5쪽 PDF](handoff/{STEM}.pdf)
- [전달 ZIP](../deliverables/OSC_SCOUT_handoff_uncapped_20261101.zip)
- [읽는 순서](handoff/START_HERE.txt)

이전 PDF·HTML·ZIP은 새 전달 자료 검증 후 제거한다. 기존 상세 문서는 이 파일로 갱신하며,
실행 근거가 되는 로그·원본 결과는 보존한다. 다른 PC 재실행용 Docker·가중치는 ZIP에 포함하지 않는다.

## 11개 항목 상태

| 번호 | 항목 | 상태 | 결과 |
|---|---|---|---|
{status_md}

## 동일 조건 비교

| 항목 | 이전 최대 10개 | 이번 개수 상한 없음 |
|---|---:|---:|
{md_rows}

`comparison.json` 및 `comparison.csv`에 비교 수치를 저장했다. 원시 검출 {det['detections']:,}개,
400장의 이미지 SHA-256, GT, 카메라 매핑, calibration/profile, 추적 parameters가 기존과 같은지 검사했다.
신뢰도 조건을 통과한 모든 검출을 선택했고 이전 상위 10개와의 정확 일치도 확인했다.
400장 중 {c['frames_exceeding_ten']}장에서 10개를 초과하며 실제 선택 수는 {c['min_selected_per_frame']}~{c['max_selected_per_frame']}개이다.

Local recall·IDF1과 multi-view IDF1은 상승했지만 Local MOTA는 하락하고 ID 전환은 증가했다.
인원 수 상한 해제만으로 추적 단절이 해결됐다고 해석하지 않는다.

## 유지한 조건과 해석

- objectness ≥0.3, 검출기 NMS와 가중치 유지. 개수 상한만 `10 → null`로 변경.
- 약 10 FPS / 5초 / GT {prep['gt_rows']:,}행·{prep['gt_ids']} ID. 첫 8카메라 주석에 보간 포함.
- 원본 frame−276=OSC frame, cam_0~7→camera_0001~8. timestamp 최대 편차 약 0.032초.
- `fps=10`, `short_track_th=40` (4초), `epsilon_scpt=.10`, `epsilon_mcpt=.37`, `time_period=3` 유지.
- 공간 거리 보정·World 기반 삭제/보간·Global 재할당 OFF. 원본 OSC 전체 설정 재현은 아님.
- GT는 시각화·평가에만 사용하고 추적 컨테이너에는 연결하지 않음.
- Local 미할당 {new['local_unassigned']:,}행, Global 보정 단계 제거 {new['global_removed']:,}행.
  raw 결과와 삭제 목록 보존. 임의 ID 생성 없음.
- cam_5에는 최종 Global 출력 없음. 카메라별 Global 행: {global_counts}.

## World 비교

예측 BBox 하단 중심 → 왜곡 보정 → ground mesh 교차를 먼저 계산했다.
이후 별도 MOT 주석 XYZ와 IoU≥0.5 일대일 매칭으로 비교했다.
좌표 상태: {world['prediction_status_counts']}.
BBox 대응 {world['bbox_matched']:,}쌍 중 World 비교 {world['world_compared']:,}쌍.
XYZ 오차 평균 / 중앙값 / p95: **{world['mean_error_dataset_units']:.3f} / {world['median_error_dataset_units']:.3f} / {world['p95_error_dataset_units']:.3f} 데이터셋 단위**.
대응된 부분집합의 사람 위치 오차이며 독립 측량에 의한 보정 오차가 아니다.
m/cm 단위·허용 오차·카메라 보정 적합성은 미확정이다. 제한 해제 전후는 대응 대상 수도 달라져
World 오차 변화만으로 보정 품질 변화를 판단하지 않는다.

## 평가·반복 검증

motmetrics 1.4.0으로 카메라별 2D와 multi-view Global identity를 분리했다.
2D는 같은 camera/frame의 IoU≥0.5, 전체 50프레임과 빈 예측을 포함한다.
Multi-view는 각 camera/frame에서만 BBox 대응하되 모든 view에 하나의 ID 대응을 계산한다.
서로 다른 카메라의 픽셀 좌표를 직접 매칭하지 않는다. World XY 진단은 별도 프로토콜이다.
공식 SCOUT/HOTA 미실시. 4개 프로토콜 self-test와 CSV 일치·중복 검사를 통과했다.

추적 2회 시간: {repeat['runs'][0]['total_seconds']:.3f}초 / {repeat['runs'][1]['total_seconds']:.3f}초.
최종 fixed_whole SHA-256: `{repeat['runs'][0]['fixed_whole_sha256']}` (두 번 동일).
신경망 추론을 재사용한 추적 시간이며 전체 FPS 또는 추론 2회 반복 결과가 아니다.

모델 커밋: `f881fe0edf60fe697efb2bf4c935b80a80861230`.
`provenance/evidence.json`에 가중치·데이터·스크립트·출력 해시, `prepare.log`, `track.log`,
`evaluate.log` 및 추론 단계 `run.log`에 Docker 이미지 ID와 실제 명령을 저장했다.

## 기존 서버에서 재실행할 경우

각 `*-new`는 존재하지 않는 새 경로를 사용한다. GPU와 데이터 경로는 실행 시 지정한다.

```bash
python3 scripts/infer-scout-sample.py \\
  --data datasets/scout/extracted/scout \\
  --adapter-export outputs/scout-first8-20s-v03-20261027/sequence1-posetrack \\
  --output outputs/osc-inference-uncapped-new --models yachiyo \\
  --gpu-posetrack 1 --gpu-others 0 --frames-per-camera 50 --max-people-per-frame 0
python3 scripts/run-osc-pilot.py prepare --output outputs/osc-pilot-uncapped-new \\
  --inference-run outputs/osc-inference-uncapped-new
python3 scripts/run-osc-pilot.py track --output outputs/osc-pilot-uncapped-new
python3 scripts/run-osc-pilot.py evaluate --output outputs/osc-pilot-uncapped-new
```

`--max-people-per-frame 0`이 개수 제한 해제이다. 과거 실험 호환을 위해 일반 스크립트 기본값 10은
유지했으며 이 실행은 명시적으로 0을 전달했다. resume 시 선택 정책이 다르면 중단한다.
자료 생성/감사 스크립트는 이번 고정 결과 경로를 사용한다.

다음 단계: 더 긴 연속 구간, Local 단절·Global 미할당 분석, World 단위·실패 처리 확정,
공간 조건 MCPT, 독립 검증 구간 및 공식 SCOUT/HOTA 평가 조건 확정.
'''
(ROOT/'docs/osc-checklist.md').write_text(details)

guide=f'''OSC 담당자 전달 자료 — 인원 제한 해제 / 2026-11-01
=================================================
결과 검토용 자료입니다. 다른 PC 재실행용 패키지가 아닙니다.

1. 먼저 볼 파일
scout-mtmc/docs/handoff/{STEM}.pdf (5쪽)
scout-mtmc/docs/handoff/{STEM}.html (이미지 내장 브라우저 열람용)
scout-mtmc/docs/osc-checklist.md (상세 검증과 기존 서버 명령)

2. 결과 폴더
scout-mtmc/outputs/osc-pilot-uncapped-20261101/
이하 경로는 위 폴더 기준입니다.

- comparison.json / comparison.csv: 이전 최대 10개 vs 제한 해제 비교.
- baseline_reference/: 비교표에 사용한 이전 평가·설정·범위 요약.
- validation_scope.json / scout_to_osc_camera_map.json: 구간과 번호 대응.
- inputs/native/: BBox·특징 경로·예측 World 좌표.
- inputs/features/: {N}개 NPY, 각각 512차원. NaN/Inf 없음.
- inputs/Pose/scene_001/: BBox–17 Keypoint JSON.
- inputs/predicted_world.json: 투영 XYZ와 성공/실패 상태.
- inputs/export/: 프레임·보정 정보 manifest/profile.
- inputs/Images/: 대표 원본 프레임 24장.
- tracking/repeat_01/fixed_camera…json: 보정 후 Local 결과.
- tracking/repeat_01/fixed_whole_tracking_results.json: 최종 Global 결과.
- tracking/repeat_01/camera…json / whole…json: 보정 전 raw 결과.
- tracking/repeat_01/settings.json / timing.json: 설정·시간.
- tracking/repeat_02/: 두 번째 실행 결과.
- tracking/repeat_report.json: 반복 동일성·시간·최종 해시.
- evaluation/local_predictions.csv: Local 할당 {L}행.
- evaluation/global_common/predictions.csv: Global {G}행 / {e['global_ids']} ID.
- evaluation/ground_truth.csv: GT {prep['gt_rows']:,}행.
- evaluation/csv_integrity.json: 일치·중복·미할당/제거 목록.
- evaluation/metrics_2d.json: 카메라별 2D 평가.
- evaluation/metrics_multiview_identity.json: 전체 view Global ID 평가.
- evaluation/metrics_world_diagnostic.json: World XY 진단.
- world_gt_comparison.json: 별도 GT World와 예측 비교.
- visualizations/gt/: GT 시각화 24장.
- evaluation/visualizations/: Local 24장 + Global 24장.
- provenance/evidence.json: 해시·소스·가중치·검증 기록.
- provenance/*.py: 실행 당시 코드 사본. 독립 실행 패키지는 아님.
- prepare.log / track.log / evaluate.log: 이미지 ID·실행 명령·로그.

scout-mtmc/outputs/osc-inference-uncapped-20261101/yachiyo/에는 원시 detections.json과
검출·Pose·ReID의 report.json/run.log를 포함했습니다. 모델 가중치, Docker 이미지,
전체 원본 영상과 외부 모델 저장소는 포함하지 않았습니다.

3. 범위와 핵심 결과
sequence_1 / cam_0~7 / 원본 frame 277~326: 카메라당 50장, 총 400장.
OSC frame = 원본 frame - 276, cam_0~7 -> camera_0001~8.
objectness >= 0.3인 {N}개를 개수 상한 없이 선택했고 원시 검출은 이전과 동일합니다.
Local recall {percent(old['local_recall'])} -> {percent(new['local_recall'])}.
Multi-view IDF1 {percent(old['multiview_idf1'])} -> {percent(new['multiview_idf1'])}.
그러나 Local MOTA {percent(old['local_mota'])} -> {percent(new['local_mota'])},
ID 전환 {old['local_id_switches']} -> {new['local_id_switches']}회로 악화된 지표도 있습니다.
Global 보정 단계 제거 {new['global_removed']:,}행, cam_5 최종 Global 출력 없음.

4. 해석 시 주의
5초 개발용 파일럿입니다. World 보정·Global 재할당 OFF, 대표 tracklet 4초 조건을
유지했습니다. 공식 OSC 전체 재현·SCOUT 벤치마크·HOTA 결과가 아닙니다.
World 오차 단위를 m/cm로 확정하지 않았습니다. GT를 추적에 주입하지 않았습니다.
GT/Local/Global의 ID 숫자는 서로 같은 번호라는 의미가 아닙니다.
기록의 /work/...는 당시 컨테이너 경로이며 위 결과 폴더에 대응합니다.
경로와 기존 기록은 증거 보존을 위해 수정하지 않았습니다.

5. 후속 논의
더 긴 구간, Local 단절·Global 미할당 원인, World 단위·실패 처리,
공간 조건을 포함한 MCPT, 독립 구간에서 공식 SCOUT/HOTA 평가.

6. 무결성
FILES.csv: 파일별 상대 경로·크기·SHA-256.
SHA256SUMS: 안내문·파일 목록을 포함한 체크섬.
압축을 푼 최상위 폴더에서 sha256sum -c SHA256SUMS로 확인할 수 있습니다.
SCOUT 일부 이미지·GT가 포함된 팀 내 인계 자료이며 공개 재배포는 이용 조건을
별도로 확인해야 합니다.
'''
(DEST/'START_HERE.txt').write_text(guide)
