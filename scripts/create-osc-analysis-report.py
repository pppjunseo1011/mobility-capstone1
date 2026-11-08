#!/usr/bin/env python3
"""Create a review report from completed OSC diagnostics and controlled experiments."""
import ast
import base64
import csv
import html
import json
from pathlib import Path

R=Path(__file__).resolve().parents[1];A=R/'outputs/osc-analysis-20261101'
D=R/'docs/handoff';D.mkdir(exist_ok=True)
S=json.loads((A/'summary.json').read_text());F=json.loads((A/'forensics5/summary.json').read_text())
rows=S['experiments'];assert len(rows)==5,'Finish four 5s settings and one 30s setting before publishing'
short=sorted([r for r in rows if r['duration_nominal_seconds']==5],key=lambda r:r['short_track_th'])
long=next(r for r in rows if r['duration_nominal_seconds']==30)
long_cam5=next(r for r in S['per_camera'] if r['duration_nominal_seconds']==30 and r['camera']=='cam_5')
long_prep=json.loads((A/'s30_th40/prepare_report.json').read_text())
baseline=next(r for r in short if r['short_track_th']==40)
cases=json.loads((A/'forensics5/failure_cases.json').read_text())
camera_metrics=list(csv.DictReader((A/'forensics5/local_stage_metrics.csv').open()))
def pct(x):return f'{100*x:.2f}%'
def num(x):return f'{x:,}'
def table(head,data):
    return '<table><thead><tr>'+''.join('<th>'+str(x)+'</th>' for x in head)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(v)+'</td>' for v in row)+'</tr>' for row in data)+'</tbody></table>'
def pic(rel,cls='plot'):
    p=A/rel;mime='image/jpeg' if p.suffix=='.jpg' else 'image/png'
    return f'<img class="{cls}" src="data:{mime};base64,{base64.b64encode(p.read_bytes()).decode()}">'
def page(n,title,body):
    return f'<section class="page"><div class="eyebrow">SCOUT × OSC · 원인 분석 및 조건별 비교</div><h1>{title}</h1>{body}<footer><span>2026.11.01 · 개발 구간 실험 · 공식 SCOUT 벤치마크 아님</span><b>{n} / 8</b></footer></section>'

pages=[page(1,'분석 결과와 핵심 판단',f'''
<p class="lead">8개 카메라의 <b>5초 실패 원인 분석 → 길이 조건 4종 비교 → 같은 시작점의 약 30초 확장</b>을 수행했습니다. 기존 결과와 원본 모델 코드를 보존했습니다.</p>
<div class="stats"><div><strong>1,826 ID</strong><span>5,329개 Local 관측의 카메라별 ID 합</span></div><div><strong>4,953 + 4</strong><span>Global 미할당 / 할당 후 삭제</span></div><div><strong>4 + 1 설정</strong><span>5초 임계값 4개 / 30초 기준 1개</span></div></div>
<h2>확인된 원인</h2>
<ul><li><b>Local 단절은 초기 SCPT부터 발생했습니다.</b> raw ID 2,029개 → 중복 억제 후 1,826개. 이번 5초 구간의 warp 분리는 추가 ID를 만들지 않았습니다.</li><li><b>단일 관측 tracklet이 1,393개(76.3%)입니다.</b> 같은 GT로 IoU 대응된 연속 관측에서 raw ID가 바뀐 1,336쌍 중 1,332쌍의 cosine similarity가 0.9 미만이었습니다. 외형 연결 문턱과 단절의 연관성이 강하지만 GT 대응 오류·검출 crop 변화도 포함될 수 있습니다.</li><li><b>Global은 대부분 대표 선택 전에 누락됩니다.</b> 길이 미달 1,789 tracklet, 길이 통과 후 Pose 탈락 24개, 통과 13개입니다.</li><li><b>short_track_th를 낮춰도 Local 결과는 같습니다.</b> 5초 Global IDF1은 12.34%에서 최대 14.80%로 올랐지만 cam_5는 네 설정 모두 출력 0행입니다.</li></ul>
<h2>30초 확장에서 새로 확인한 사항</h2>
<p>원본 frame 277~576, 카메라당 300장입니다. 같은 frame 번호의 카메라 간 timestamp 차이가 최대 <b>{S['max_timestamp_skew_seconds']:.3f}초</b>이며 일부 시간 간격이 불연속입니다. 같은 설정의 진단 실험으로 기록했고 시간 동기화 완료로 표시하지 않았습니다.</p>
<div class="callout">HOTA·DetA·AssA는 이번에 계산하지 않았습니다. 2D Local 평가와 view별 BBox 매칭을 사용하는 Global IDF1을 구분했고, 공식 평가 프로토콜 확정은 다음 단계입니다.</div>
''')]

camtable=[]
for c,v in F['camera_summary'].items():
    m=next(r for r in camera_metrics if r['camera']==c and r['stage']=='after_warp')
    camtable.append((c,v['tracklets'],v['length_bins'].get('1',0),int(float(m['num_switches'])),pct(float(m['idf1']))))
pages.append(page(2,'Local ID 단절: 단계별 근거',f'''
<p>1,826은 사람 수가 아니라 <b>카메라별 Local ID 개수의 합</b>입니다. 같은 구간 GT의 camera–ID 조합은 {S['five_second_gt_camera_id_pairs']}개입니다. 동일 인물의 ID 분할뿐 아니라 오검출의 독립 ID도 포함될 수 있습니다.</p>
{table(['카메라','Local ID 수','길이 1','IDSW','Local IDF1'],camtable)}
{pic('figures/tracklet_lengths.png')}
<p class="small">길이는 시간 span이 아닌 관측 행 수입니다. tracklets.csv에 관측 수·span·시작/끝 frame을 모두 보존했습니다. 카메라별 IDSW 합은 1,328입니다.</p>
<h2>분리한 처리 단계</h2>
<p><b>raw SCPT 2,029 ID → sequential NMS 1,826 ID → warp 1,826 ID.</b> NMS에서 240행이 Local 미할당으로 바뀌었습니다. 원본 보정 함수를 다시 적용한 결과가 저장된 fixed JSON과 정확히 일치했습니다.</p>
<div class="note">원인 후보는 엄격한 외형 유사도 문턱, 같은 프레임 중복 클러스터 분리, 인접 3초 구간 연결 실패입니다. warp 분리가 이 5초 결과의 주원인이라는 가설은 재현 결과로 지지되지 않았습니다.</div>
'''))

chosen=[next(c for c in cases if c['camera']==n) for n in ('cam_5','cam_7')]
cards=''.join('<div style="width:48%"><h3>'+c['camera']+f' · GT {c["gt_id"]}</h3><p class="small">frame {c["previous_source"]} → {c["source_frame"]}<br>Local {c["previous_local_id"]} → {c["new_local_id"]}</p>'+pic('forensics5/'+c['image'],'case')+'</div>' for c in chosen)
pages.append(page(3,'실패 사례와 외형 특징 진단',f'''
<p>초록색은 해당 GT BBox, 빨간색은 매칭된 예측 BBox입니다. 위/아래 영상은 IDSW 전후이며, 프레임·GT·Local ID를 이미지에 표시했습니다.</p>
<div style="display:flex;justify-content:space-between">{cards}</div>
{pic('figures/appearance_gate.png')}
<p class="small">GT IoU≥0.5 Hungarian 대응으로 얻은 연속 관측 3,776쌍 중 1,587쌍(42.0%)의 cosine이 0.9 미만입니다. raw ID 변경 쌍 중 99.7%가 이 범위에 속합니다. 이 대응은 추적 입력이나 성능 계산을 대체하지 않는 진단입니다. 원본 SCPT는 float16 유사도를 사용하므로 문턱 부근의 반올림 차이도 존재합니다.</p>
<div class="note">다음 Local 개선 실험은 더 긴 GT track의 crop·유사도 확인 후 epsilon_scpt를 하나씩 비교하는 방식이 적절합니다. 이번에는 요청한 short_track_th만 바꿨으며 ReID 재학습이나 임계값 동시 변경은 하지 않았습니다.</div>
'''))

gatetable=[]
for c,v in F['camera_summary'].items():
    g=v['sequential_rejection'];gatetable.append((c,v['tracklets'],g.get('length',0),g.get('pose',0),g.get('eligible',0)))
pages.append(page(4,'Global 누락: 4,953행과 4행',f'''
<div class="callout"><b>5,569 입력 = 4,953 미할당 + 616 할당</b><br><b>616 할당 − 4 Pose 후처리 삭제 = 최종 612행</b></div>
{table(['단계별 원인','관측 행 수'],[('Local ID 미할당 (sequential NMS)',240),('Local은 있으나 대표 tracklet 길이 미달',3548),('길이 통과 후 대표 Pose 조건 탈락',1165),('Global 할당 후 Pose 노이즈 삭제',4)])}
<h2>tracklet 기준 탈락 내역 (short_track_th=40)</h2>
{table(['카메라','전체','길이 미달','길이 통과·Pose 탈락','최종 통과'],gatetable)}
<p class="small">순서대로 분류한 상호 배타적인 수입니다. 짧으면서 Pose도 나쁜 tracklet은 길이 미달에 먼저 포함됩니다. 두 조건의 교차 분포도 summary.json과 tracklets.csv에 따로 보존했습니다.</p>
<h2>cam_5가 비는 이유</h2>
<p>199 tracklet의 최대 길이는 <b>38</b>이므로 th=40에서는 모두 탈락합니다. th=10·20·30에서도 길이를 통과한 후보가 Pose 조건을 충족하지 못해 최종 Global 출력이 없습니다. Pose를 통과한 짧은 후보는 길이 10 미만입니다.</p>
<p class="small">삭제된 4행은 모두 cam_4 / Local ID 0 / Global ID 7입니다. source frame 277·282·283은 confidence &gt;0.75인 Keypoint가 0개, frame 291은 1개여서 삭제됐습니다. 해당 serial·이유는 global_row_exclusions.csv에 기록했습니다.</p>
'''))

experiment_table=[(r['short_track_th'],pct(r['local_idf1']),r['local_idsw'],pct(r['global_idf1_multiview']),r['final_global_rows'],r['global_ids']) for r in short]
pages.append(page(5,'한 조건만 변경한 5초 비교',f'''
<p><b>고정:</b> source frame 277~326, 8카메라, 인원 상한 없음, objectness≥0.3, 동일 검출·Pose·임베딩·가중치, epsilon_scpt=0.10, epsilon_mcpt=0.37, 공간 보정·Global 재할당 OFF.</p>
{table(['short_track_th','Local IDF1','Local IDSW','Global IDF1','Global 행','Global ID'],experiment_table)}
{pic('figures/threshold_sweep.png')}
<h2>변화의 의미</h2>
<p class="small">길이·Pose 조건을 모두 통과한 대표 tracklet은 th=40 / 30 / 20 / 10에서 각각 13 / 14 / 17 / 18개입니다. 길이 조건을 완화해도 다수 후보는 Pose 조건에서 탈락합니다.</p>
<ul><li>네 설정의 fixed Local JSON은 기준 결과와 <b>바이트 단위로 동일</b>했습니다. Local IDF1·IDSW가 변하지 않는 것은 정상입니다.</li><li>th=40 결과는 기존 실행과 최종 Global JSON·평가 수치가 같았습니다. 각 설정을 2회 실행해 최종 해시 동일성도 확인했습니다.</li><li>th=10에서 Global 행은 123행 증가하고 IDF1은 약 2.46%p 상승했습니다. 그래도 전체 5,569개 입력 중 735행만 남아 누락 문제가 큽니다.</li><li>이 개발 구간에서 좋은 값이 일반적인 최적값이라는 뜻은 아닙니다. 별도 검증 구간에서 재확인이 필요합니다.</li></ul>
<p class="small">Global IDF1은 view별 BBox 대응 후 모든 카메라의 ID를 통합한 지표입니다. 카메라별 Global 2D IDF1은 CSV에 별도 열로 기록했습니다. IDSW는 카메라별 Local 합계이며 임의 view 직렬화의 Global IDSW를 보고하지 않습니다.</p>
'''))

longtable=[('source frame','277~326','277~576'),('카메라당 프레임','50','300'),('short_track_th',40,40),
('Local IDF1',pct(baseline['local_idf1']),pct(long['local_idf1'])),('Local IDSW',baseline['local_idsw'],long['local_idsw']),
('Global IDF1',pct(baseline['global_idf1_multiview']),pct(long['global_idf1_multiview'])),
('Global 행 / ID',f'{baseline["final_global_rows"]} / {baseline["global_ids"]}',f'{long["final_global_rows"]} / {long["global_ids"]}')]
pages.append(page(6,'약 30초 연속 구간 확장',f'''
{table(['항목','기존 5초','확장 약 30초'],longtable)}
<p class="small">확장 구간 입력 {long_prep['native_prediction_rows']:,}개, GT {long_prep['gt_rows']:,}행·{long_prep['gt_ids']} ID. cam_5에도 최종 Global {long_cam5['global_rows']}행이 생성됐습니다. 다른 사람이 추가로 등장하는 구간이므로 짧은 구간과 성능을 단순 우열 비교하지 않습니다.</p>
{pic('figures/timestamp_skew.png')}
<h2>시간 정보를 함께 해석해야 합니다</h2>
<ul><li>300프레임은 nominal 10 FPS 기준 약 30초입니다. 카메라별 첫/끝 timestamp 경과는 약 29.882~30.590초입니다.</li><li>cam_0의 563→564는 1.273초, cam_3의 566→567은 3.400초, cam_4의 500→501은 0.299초 간격입니다. 같은 frame의 최대 카메라 차이는 2.051초입니다.</li><li>카메라별 2D 평가와 view별 BBox 대응의 Global identity 진단은 기록하되, 동일 frame의 World 좌표 융합은 시간 정렬이 검증된 결과로 해석하지 않습니다.</li><li>5초는 이 30초 구간의 앞부분입니다. 서로 독립된 test split이 아니며 긴 구간의 IDSW 절대 횟수를 짧은 구간과 직접 우열 비교하지 않습니다.</li></ul>
<div class="note">30초 실험은 기존 th=40을 유지했습니다. 5초에서 수치가 오른 th=10을 동시에 적용하지 않아 구간 길이와 임계값의 영향을 분리했습니다.</div>
'''))

code_rows=[
('tracking.py:106–139','Tracker.scpt','frame // (time_period×fps)로 구간 분할, 구간 cluster에 새 ID offset 부여, 직전 구간과 연결. 1-based frame에서 첫 구간은 1~29.'),
('scpt.py:321–330','create_similarity_matrix_scpt','cosine을 float16으로 바꾼 뒤 similarity &lt; 1−epsilon_scpt이면 0. 현재 문턱 0.9.'),
('scpt.py:334–365','tracking_by_clustering','외형 계층 클러스터링 후 같은 frame 중첩 cluster를 다시 분리.'),
('scpt.py:13–113, 369–394','create_centrality_matrix / associate_cluster_between_period','같은 frame이 겹치는 cluster는 병합 금지. similarity &gt; 1−epsilon인 연결의 합이 문턱을 넘어야 병합.'),
('scpt.py:407–465, 505–542','sequential_non_maximum_suppression / separate_warp_tracklet','시간·공간 overlap 기준 0.6으로 중복 관측을 -1 처리. warp_th=40 초과시 새 ID 생성; 이번 5초에서는 추가 분리 없음.'),
('mcpt.py:32–56, 299–322','create_similarity_matrix_mcpt / create_camera_dict','len(all_serials)&lt;short_track_th 또는 대표 score&gt;keypoint_condition_th이면 MCPT 대상 제외.'),
('mcpt.py:67–142','eval_keypoints / find_high_confidence_keypoint_node','condition 1은 17개 Keypoint 모두 기준 이상. 후보 중 가장 좋은 condition, 그중 가장 큰 BBox를 대표로 선택.'),
('mcpt.py:240–258','multi_camera_people_tracking','대표 대상 local_id에만 GlobalOfflineID를 부여. 재할당 OFF이면 나머지를 복구하지 않음.'),
('mcpt.py:708–766','eval_noise_level / remove_noise_images','Global 키 없는 행 제거 후 Pose 누락·비정상 종횡비·낮은 confidence Keypoint 수 조건으로 추가 삭제.')]
with (A/'code_findings.csv').open('w') as stream:
    writer=csv.writer(stream);writer.writerow(['source_file_and_lines','function','condition_and_effect'])
    writer.writerows((p,fn,html.unescape(condition)) for p,fn,condition in code_rows)
pages.append(page(7,'관련 코드와 실제 조건문',f'''
<p class="small">경로 기준: <code>repos/yachiyo/tracking/src/</code><br>원본 커밋: <code>f881fe0edf60fe697efb2bf4c935b80a80861230</code></p>
{table(['파일·행 / 함수','조건과 영향'],[(f'<b>{p}</b><br>{fn}',condition) for p,fn,condition in code_rows])}
<div class="note"><b>원본 코드에서 확인한 전달 누락:</b> scpt.py:349는 <code>epsion=epsilon</code> 오타로 초기 계층 클러스터링의 기본 epsilon=0.3을 사용합니다. mcpt.py:130은 keypoint_th를 eval_keypoints에 전달하지 않아 실제 기준은 기본값 0.7입니다(설정 0.8과 다름). 이번 비교에서는 원본 동작을 보존했습니다. 이 둘을 Local 단절의 단독 원인으로 확정하지 않았습니다.</div>
'''))

pages.append(page(8,'전달 파일과 다음 검증 단계',f'''
<p>ZIP은 결과 검토용입니다. 모델 가중치·Docker 이미지·전체 SCOUT 원본은 제외했습니다. <b>START_HERE.txt</b>에서 파일별 용도와 경로를 확인할 수 있습니다.</p>
{table(['산출물','파일/폴더'],[('전체 비교·카메라별 지표','experiment_comparison.csv · per_camera_comparison.csv'),('길이·Pose 탈락 상세','tracklet_gate_comparison.csv · forensics5/tracklets.csv'),('단계별 Local 원인','forensics5/local_stage_counts.csv · local_stage_metrics.csv'),('4,953 미할당 / 4 삭제','forensics5/global_row_exclusions.csv'),('실패 사례·외형 진단','forensics5/failure_*.jpg · idsw_events.csv<br>adjacent_gt_appearance.csv · figures/'),('각 설정 결과','s5_th10/20/30/40, s30_th40 아래 tracking/ · evaluation/'),('시간 동기화 확인','timestamp_skew.csv · summary.json'),('코드 수정 전후·증거','provenance/runner_changes.patch · original_*.py<br>provenance/evidence.json · 실행 코드 사본')])}
<h2>변경한 코드의 범위</h2>
<p>실험 runner에 길이 파라미터와 300프레임 처리, 가변 범위 평가, IDSW 이벤트 저장을 추가했습니다. 원본 OSC 모델 소스는 수정하지 않았습니다. 원본과 설정 변경 실험을 혼합하지 않도록 코드 패치와 실제 파라미터·이미지 ID·해시를 보존했습니다.</p>
<h2>모델 간 공통 평가를 위한 다음 단계</h2>
<ol><li>timestamp 불연속·카메라 시간 대응을 확인하고 공통 검증 구간을 고정합니다.</li><li>Local 외형 연결과 Pose 대표 조건을 각각 별도 실험으로 검증합니다.</li><li>GT 대상, ignore 처리, ID 범위, frame 대응과 IoU 규칙을 세 모델에 동일하게 적용합니다.</li><li>HOTA·DetA·AssA를 정해진 평가 구현으로 추가하고, IDF1·IDSW의 2D/Global 범위를 명확히 구분합니다.</li><li>개발 구간에서 선택한 설정을 독립 검증 구간에서 재평가합니다.</li></ol>
<p class="small">이번 자료에서 HOTA·DetA·AssA는 빈 값이며 0점이 아닙니다. World 진단, 카메라별 2D, multi-view identity를 한 지표로 섞지 않습니다.</p>
'''))

tree=ast.parse((R/'scripts/create-osc-handoff.py').read_text())
css=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='css' for t in n.targets))
css+='\n.plot{width:100%;height:auto;display:block;margin:3mm 0;}.case{width:100%;height:auto;} table{font-size:8.4pt;} td{padding:1.7mm 2mm;} .page:nth-child(7) table{font-size:7.8pt;line-height:1.45;}.page:nth-child(7) td:first-child{width:40%;overflow-wrap:anywhere;}'
out=D/'OSC_SCOUT_원인분석_비교실험_20261101.html'
out.write_text('<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8"><title>SCOUT OSC 원인 분석 및 비교 실험</title><style>'+css+'</style></head><body>'+''.join(pages)+'</body></html>')
print(out)
