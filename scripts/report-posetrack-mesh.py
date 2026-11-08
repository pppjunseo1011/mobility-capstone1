#!/usr/bin/env python3
"""Create a self-contained Korean HTML report from completed experiment files."""
import base64,csv,html,json
from pathlib import Path
R=Path(__file__).resolve().parents[1];O=R/'outputs/posetrack-mesh-20261104';D=R/'docs/handoff'
def load(p):return json.loads(p.read_text())
def table(headers,rows):return '<table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in r)+'</tr>' for r in rows)+'</tbody></table>'
def img(name,caption):
 p=O/'visualizations'/name;mime='image/png' if p.suffix=='.png' else 'image/jpeg'
 return '<figure><img src="data:'+mime+';base64,'+base64.b64encode(p.read_bytes()).decode()+'"><figcaption>'+caption+'</figcaption></figure>'
def main():
 D.mkdir(exist_ok=True);find=load(O/'findings.json');audit=load(O/'execution_audit.json');runs={r['run']:r for r in find['experiment_rows']}
 assert 'C_30s' in runs
 b,c,long=[runs[n] for n in ['B_final','C_final','C_30s']];pages=[]
 def page(title,body):pages.append('<section><div class="eyebrow">SCOUT × PoseTrack · 2026.11.04 · '+str(len(pages)+1)+'/8</div><h1>'+title+'</h1>'+body+'</section>')
 names={'B_final':'B · 지면 항 제외','C_final':'C · Mesh 적용','C_30s':'C · 30초 확장'}
 summary=table(['실험','입력 관측','출력 행','Global ID','입력 대비 출력'],[[names[n],f"{runs[n]['input_rows']:,}",f"{runs[n]['output_rows']:,}",runs[n]['global_ids'],f"{100*runs[n]['output_fraction']:.2f}%"] for n in names])
 page('전체 영역 Tracking 실행 결과',f'''<p class="lead">평면 ROI 제한을 제거하고 Sequence 1의 cam_0~cam_7 전체 영상 영역에서 Tracking을 실행했다. 5초 입력 5,569건을 모두 처리했으며, Mesh 기반 기하를 적용한 30초 확장도 완료했다.</p>
 <div class="callout">기존 TITAN RTX 서버와 기존 Docker 이미지를 사용했다. H200 MIG 서버로 이동하거나 새 GPU 환경을 구성하지 않았다.</div>
 {summary}<p>한 행은 한 카메라의 한 프레임에서 관측된 사람 한 명을 뜻한다. Global ID 수는 서로 다른 <b>예측 ID의 수</b>이며 실제 인원 수가 아니다. 출력 비율은 추적 정확도가 아니다.</p>
 <h2>실험 범위</h2><ul><li>A: 기존 평면 ROI 444건 → 416행·6 ID. 원본 비계측 실행과 출력 및 프레임별 상태·Feature Bank 개수가 동일했다.</li><li>B/C: source frame 277~326, 카메라당 50프레임, 총 400장. 동일 Detection·Pose·Re-ID를 재사용했다.</li><li>30초: source frame 277~576, 카메라당 300프레임, 총 2,400장. 기존 5초를 그대로 유지하고 나머지 2,000장의 Pose·Re-ID만 추가 생성했다.</li></ul>
 <h2>이번에 확인한 범위</h2><p>검출 → 포즈 → 특징 → 기하 점수 → Hungarian 배정 → Global ID 출력 연결, 입력·출력 BBox 대응, 내부 로그 저장, GT 사후 평가를 확인했다. 장면 전체의 물리적 보정 정확도와 일반적인 추적 성능을 입증한 결과는 아니다.</p>''')
 q=find['geometry_discrepancy_xy']
 page('최초 원인: 단일 평면과 실제 지면의 차이',f'''<p>SCOUT는 렌즈 왜곡 계수와 높이가 변하는 지면 Mesh를 제공한다. 기존 PoseTrack의 P에는 렌즈 왜곡이 없고, H는 하나의 평면을 가정한다. 행렬 모양이나 가역성만으로 좌표 변환이 일치하지 않는다.</p>
 <ol><li><b>렌즈 왜곡:</b> 원본 검출·관절 좌표를 왜곡 보정한 뒤 P=K[R|t]와 결합했다. 추적 결과를 평가할 때는 원본 관측 BBox를 다시 찾아 사용했다.</li><li><b>높이 기준:</b> 원본의 절대 Z 제한을 지면 대비 높이 제한으로 바꿨다. Mesh 전체 Z 범위는 약 −4.345~0.849이다. 단순 Z 원점 이동만으로 지면 전체를 평면으로 만들 수 없다.</li><li><b>지면 위치:</b> Z=0 평면 역투영 대신 광선–Mesh 교차점을 사용한다. 발목 위치와 지면 접점도 구분한다.</li></ol>
 {img('geometry_discrepancy.png','왼쪽: Mesh 접점과 Z=0 평면 추정점의 차이. 오른쪽: 동일 영상 위치의 두 역투영 결과 차이. GT 기준 위치 오차가 아니다.')}
 <p>5초 BBox 접점 5,015건에서 XY 차이 중앙값 {q['median']:.3f}, 95백분위 {q['p95']:.3f}, 최대 {q['max']:.3f} SCOUT 좌표 단위. 554건은 BBox 광선이 Mesh와 만나지 않지만 입력에서 제거하지 않았다.</p>
 <p class="small">Mesh 전체에 최적의 단일 경사 평면을 맞춰도 수직 거리 기준 95백분위 잔차는 {audit['global_plane_fit']['orthogonal_residual_p95']:.3f}, 최대 {audit['global_plane_fit']['orthogonal_residual_max']:.3f}이다. 단일 평면과 정확히 일치하지 않음을 확인했다. K/R/t는 SCOUT 정의를 사용하며 카메라 중심은 −Rᵀt이다. 수치적 일치 검증이며 독립 실측이나 스케일 재측량 검증은 아니다.</p>''')
 page('수정 내용과 통제한 조건','''<p>원본 저장소는 수정하지 않았다. 별도 실행 스크립트가 원본 일치 여부를 확인한 뒤 패치 사본을 만들고, 각 실험 폴더에 diff와 실제 실행 소스를 저장한다.</p>'''+table(['위치 / 함수','수정 또는 유지한 내용'],[
 ['prepare-posetrack-mesh.py','17개 관절·BBox 경계 왜곡 보정. 원본 BBox 하단 중심 광선 별도 보존. 모든 입력 관측 유지.'],
 ['PoseTracker.compute_epi_homo_aff → MeshGeometry.affinity','Epipolar 계산은 원본 수식 유지. C의 Ah는 Mesh 접점 사이 3D 거리로 대체. B는 Ah와 a_homo를 모두 0 처리.'],
 ['PoseTracker.target_init → init_affinities','새 ID 초기화 단계에도 동일한 Mesh/제외 조건 적용. 원본 2×Aepi+Ah 클러스터링 점수 유지.'],
 ['PoseTrack.multi_view_3D_update / get_output','절대 Z 대신 해당 XY의 Mesh 지면 대비 높이로 검사. 출력은 기존 삼각측량 우선순위 후 Mesh 투영 사용. B/C 공통.'],
 ['PoseTrack.switch_view','NumPy 행 뷰 교환 시 .copy() 적용. 기존 대입은 BBox·관절·age 행을 덮어쓸 수 있어 관측 ID와 BBox가 어긋남. B/C 공통 수정.'],
 ['GLPKSolver._solve_bip','B에서 초기 정수계획 지연 발생. B/C 모두 같은 목적함수·제약의 SciPy HiGHS 사용. 최적해 status=0만 허용. C의 5초 결과는 기존 GLPK와 바이트 단위 동일.']])+'''
 <h2>Mesh 점수와 발목 처리</h2><p><code>ray = normalize((KR)⁻¹ [u,v,1]ᵀ)</code><br><code>Ah = mean(1 − ||ground_i − ground_j||₂ / 1.5)</code></p>
 <p>양쪽 발목 confidence &gt;0.7일 때 Mesh를 월드 Z 방향 0.15만큼 올린 표면과 교차시킨 후 지면 접점을 구한다. 0.15는 기존 높이 가정을 이어받은 값이며, 실제 발목 높이를 측정한 값이 아니다. 발목 교차 실패 시 BBox 접점으로 대체하고, 이것도 실패하면 해당 지면 항만 중립 처리한다.</p>
 <p>Epipolar 거리 기준 0.2, 지면 거리 기준 1.5, 상태·매칭·특징 저장 임계값은 원래 숫자를 유지했다. 거리 단위는 SCOUT 월드 좌표이며 단위/분포 적합성의 최적화는 하지 않았다. B는 원본 전체 기능을 검증하는 실험이 아닌 지면 항 진단 실험이다.</p>''')
 metric_rows=[]
 for key,label in [('HOTA_2D','HOTA (2D)'),('DetA_2D','DetA (2D)'),('AssA_2D','AssA (2D)'),('IDF1_2D','IDF1 (카메라별 합산)'),('IDF1_view_aware','IDF1 (공유 Global ID)'),('IDSW_2D','IDSW (카메라별 합)')]:
  metric_rows.append([label,str(b[key]) if key=='IDSW_2D' else f'{100*b[key]:.2f}',str(c[key]) if key=='IDSW_2D' else f'{100*c[key]:.2f}'])
 camrows=list(csv.DictReader((O/'C_final/per_camera.csv').open()))
 page('5초 비교: 동일 입력에서 지면 항만 비교',table(['지표 · IDSW 외 0~100','B · 지면 항 제외','C · Mesh 적용'],metric_rows)+
 f'''<p>이 5초 구간에서는 Mesh 적용 후 HOTA가 {100*b['HOTA_2D']:.2f}→{100*c['HOTA_2D']:.2f}, IDSW가 {b['IDSW_2D']}→{c['IDSW_2D']}로 바뀌었다. 다만 출력 비율은 {100*b['output_fraction']:.2f}%→{100*c['output_fraction']:.2f}%로 감소했다. 결과를 모든 구간의 성능 향상으로 일반화하지 않는다.</p><h2>C의 카메라별 처리 결과</h2>'''+
 table(['카메라','입력','출력','예측 Global ID'],[[r['camera'],r['input_observations'],r['output_rows'],r['predicted_global_ids']] for r in camrows])+'''
 <p class="small">유효 World XY가 없는 출력은 B 153행, C 146행이다. 2D 평가는 이 행도 포함한다. GT는 전체 7,589개 주석을 사후 평가에만 사용했다. TrackEval: HOTA는 IoU 0.05~0.95 평균, IDF1/IDSW는 IoU 0.5. 2D는 카메라별 합산, 공유 Global ID 평가는 카메라·프레임 관측을 분리한 별도 프로토콜이다. 공식 SCOUT 3D 벤치마크가 아니다. A는 ROI와 GT 범위가 달라 이 표에서 제외했다.</p>''')
 la=audit['C_30s'];ts=audit['time_alignment']['300'];s=load(O/'C_30s/summary.json')
 page('30초 확장과 아직 남은 한계',table(['항목','C · Mesh / 300프레임'],[
 ['입력 / 출력 / Global ID',f"{long['input_rows']:,} / {long['output_rows']:,} / {long['global_ids']}"],
 ['HOTA / DetA / AssA (2D)',f"{100*long['HOTA_2D']:.2f} / {100*long['DetA_2D']:.2f} / {100*long['AssA_2D']:.2f}"],
 ['IDF1 (2D / 공유 Global ID)',f"{100*long['IDF1_2D']:.2f} / {100*long['IDF1_view_aware']:.2f}"],
 ['IDSW',long['IDSW_2D']],['출력 비율',f"{100*long['output_fraction']:.2f}%"],
 ['유효 World XY 없는 출력 행',long['missing_world_rows']],
 ['Missing Re-ID 결정 / 수락',f"{la['event_counts'].get('missing_reid_decision',0)} / {la['accepted_missing_reid_decisions']}"],
 ['재활성화 / view 교환 이벤트',f"{la['event_counts'].get('reactivation',0)} / {la['event_counts'].get('switch_view',0)}"],
 ['계측 포함 Tracking 시간',f"{s['elapsed_seconds']:.1f}초 (CPU; 전체 파이프라인 속도 아님)"]])+f'''
 <div class="callout">프레임 번호 기준 300프레임을 nominal 30초로 사용했다. 카메라 간 timestamp 차이는 최대 {ts['max_cross_camera_skew_seconds']:.3f}초이고, 300개 시점 중 {ts['frames_with_skew_over_50ms']}개가 50ms를 초과한다. 5초와 30초의 차이를 Mesh 효과만으로 해석하면 안 된다.</div>
 <p>추가 구간의 Pose 31,454건과 2,048차원 Re-ID 특징을 GPU 1에서 생성했다. 가중치 로드, finite/nonzero 검증을 통과했다. 원래 5초 관측의 순서·특징을 유지했다. 30초 출력 중 {long['missing_world_rows']}행은 삼각측량·Mesh 접점으로 유효 World XY를 얻지 못했다. 이 행들의 2D BBox와 ID는 유지하고 위치는 미확정으로 기록했다. 임의 좌표를 넣지 않았다.</p>
 <ul><li>10FPS 입력을 사용하되 원본 프레임 기준 age/유지 임계값을 변경하지 않았다. 시간 단위 재조정은 후속 비교 항목이다.</li><li>30초에서는 C만 실행했다. B/C 장시간 성능 비교 실험은 아직 하지 않았다.</li><li>재활성화·교환 로그는 알고리즘이 실행됐다는 증거다. 해당 동작의 GT 기준 정답률은 이번에 별도 분석하지 않았다.</li><li>가려짐·검출 누락·ID 혼합이 남아 있다. Mesh 누락 영역, 수직 겹침 표면 선택, 발목 0.15 가정, 비동기 프레임 정렬을 추가 검토해야 한다.</li></ul>''')
 page('예측 Global ID 연결 사례',img('cross_camera_detail.jpg','실제 추적 결과의 동일 예측 Global ID를 카메라 간 표시했다. GT ID로 예측을 치환하지 않았다. 선정 단계에서만 IoU≥0.5 사후 GT 대응을 참고했다.')+
 img('suspected_id_merge_detail.jpg','동일 예측 ID가 서로 다른 GT 사람과 대응된 의심 사례. GT 대응은 사후 IoU 매칭이므로 개별 오류의 확정 판정에는 영상 확인이 필요하다.')+
 '<p class="small">원본 해상도 이미지는 visualizations 폴더에 있으며, cam_0~7 각각의 frame 302 전체 예측 BBox 이미지도 포함했다. 표시된 PRED ID는 모델이 출력한 번호다.</p>')
 page('내부 계산과 로그 읽는 법',table(['파일','내용'],[
 ['predictions_raw.csv','원본 영상 좌표 BBox, camera, source_frame, model_frame, global_id, confidence, observation_id. local_id는 별도 출력이 없어 공란.'],
 ['affinity_pairs.csv','Ab/Aepi/Ah/As, final_score, 가중치, norm, Re-ID 활성화 여부, Hungarian 수락 결과.'],
 ['matrices/frame_…_cam_….npz','각 카메라/프레임의 Affinity 행렬. events.json의 track_ids/observation_ids로 행·열 식별.'],
 ['events.json / state_history.csv','Hungarian 후보 및 수락·거절 사유, 초기화, Detected/Occluded/Missing, Re-ID 결정 및 ID 교환.'],
 ['feature_bank_updates.npz','변경된 특징 벡터. events.json의 feature_bank/vector_keys와 연결. 저장 거절 조건은 feature_bank_gate 참조.'],
 ['geometry_observations.csv / mesh_events.json','BBox·발목 교차점/교차 실패 및 관절 높이 제한 검사. geometry_details.json은 예전 형식의 빈 파일이므로 제외.'],
 ['initialization_affinity_frame_….npy / solver.jsonl','새 ID 초기화의 클러스터링 행렬과 정수계획 최적해 상태.'],
 ['configuration.json / native_changes.patch','수치 조건, 구현 차이 및 원본 대비 diff.'],
 ['metrics.json / protocol.json','평가 요약·상세 및 프로토콜. posthoc_gt_matches.csv는 평가용 대응이다.']])+'''
 <h2>최종 Affinity 수식 점검</h2><p><code>S = (Aepi + 5·Ab + a_homo·Ah + 5·As·w) / (1+w)</code><br><code>norm = 1[Aepi≠0] + 5·1[Ab≠0] + a_homo·1[Ah≠0]</code><br>norm&gt;0이면 S에서 (5−norm)×0.1을 빼고, 마지막으로 0 미만을 0으로 제한한다. C의 a_homo=1, B는 0이다. w는 원본 상태 기반 Re-ID 가중치다.</p>
 <p>모든 저장 pair에서 위 수식과 최종 점수를 재계산해 일치 여부를 검사했다. Epipolar 구현은 원본 C++와 최대 절대차 약 2.84×10⁻¹⁴였다. 이 검증은 수식 일치 검증이며 매칭 정답 여부를 보장하지 않는다.</p>''')
 page('재현 및 전달 범위','''<p>결과 ZIP은 보고서·실험 결과·분석 로그·패치·실행 스크립트·입력 매핑·검증 내역을 포함한다. 데이터셋 전체, 모델 가중치, Docker 이미지, 원본 저장소 전체는 포함하지 않는다.</p>
 <h2>현재 서버에서 결과 확인</h2><pre>outputs/posetrack-mesh-20261104/
  A_replay/verification.json
  B_final/   C_final/   C_30s/
  full_input/   full_input30/
  experiment_comparison.csv
  execution_audit.json
  geometry_verification.json
  reproducibility/
  visualizations/</pre>
 <h2>재실행</h2><p>프로젝트 루트에서 <code>scripts/replay-posetrack-mesh.sh</code>에 새 결과 폴더를 지정한다. 데이터 경로와 기존 입력 경로는 환경 변수로 지정할 수 있다. 기존 실험 폴더에는 덮어쓰지 않는다. 내부 실행 명령·실제 수행한 단계는 <code>reproducibility/commands.txt</code>를 참고한다.</p>
 <h2>환경 및 원본 보존</h2><p>PoseTrack master ba727e6bc6e45e44cd6e006b73e00864892173c2.<br>이미지 scout-mtmc/posetrack-all:ba727e6, Python 3.9 / PyTorch 2.0.1+cu118 / NumPy 1.23.5. 추적·평가는 CPU, 추가 Pose/Re-ID만 TITAN RTX GPU 1에서 실행했다. 새 패치와 확장 스크립트는 원본 저장소 밖에 저장했다.</p>
 <h2>후속 작업의 우선순위</h2><ol><li>30초의 실제 timestamp에 맞춰 카메라 관측을 정렬하고 재평가.</li><li>Mesh 미교차·발목 높이 가정 및 ID 혼합 사례를 영상과 로그로 점검.</li><li>상태 기반 Re-ID와 Feature Bank를 개별 분석. 숫자 임계값과 시간 단위 조정은 하나씩 변경.</li></ol>
 <p class="small">결론: 전체 영역 Tracking과 Mesh 경로 실행은 완료했고 이 5초 구간의 B/C 차이를 평가했다. 데이터셋 전체에 대한 정확한 3D 보정, 논문 성능 재현 또는 모델 간 공정 비교가 완료된 것은 아니다.</p>''')
 css='''@page{size:A4;margin:15mm 14mm}*{box-sizing:border-box}body{font-family:"Noto Sans CJK KR",sans-serif;font-size:10pt;color:#1c2735;margin:0;line-height:1.6}section{break-after:page}section:last-child{break-after:auto}.eyebrow{font-size:8pt;color:#63788d;letter-spacing:.3px;border-bottom:2px solid #2c6689;padding-bottom:5px}h1{font-size:21pt;line-height:1.3;margin:15px 0 16px;color:#123e59}h2{font-size:12pt;margin:17px 0 7px;color:#245c78}p{margin:9px 0}.lead{font-size:12pt}table{width:100%;border-collapse:collapse;font-size:9pt;line-height:1.5;margin:12px 0;table-layout:fixed}td,th{padding:7px 8px;border-bottom:1px solid #d4dde4;overflow-wrap:anywhere;vertical-align:top}th{background:#1e526e;color:white;text-align:left}tr:nth-child(even){background:#f2f6f8}.callout{padding:11px;background:#eaf3f7;border-left:4px solid #347c9f;margin:12px 0}.small,figcaption{font-size:8.3pt;color:#526375}ul,ol{padding-left:22px}li{margin:6px 0}figure{margin:12px 0}figure img{width:100%;max-height:86mm;object-fit:contain}code,pre{font-family:"DejaVu Sans Mono",monospace;font-size:8pt;overflow-wrap:anywhere}pre{padding:10px;background:#f1f4f6;white-space:pre-wrap}'''
 text='<!DOCTYPE html><html lang="ko"><meta charset="utf-8"><title>PoseTrack SCOUT 전체 영역 Mesh 검증</title><style>'+css+'</style><body>'+''.join(pages)+'</body></html>'
 (D/'PoseTrack_SCOUT_전체영역_Mesh_검증_20261104.html').write_text(text)
 print('REPORT HTML',len(pages),'sections')
if __name__=='__main__':main()
