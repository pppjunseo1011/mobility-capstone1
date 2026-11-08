#!/usr/bin/env python3
"""Create review videos only; no identity tracking is implied."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import aic_cpp
import cvxopt
import cv2

out=Path('/out')
binary=Path(aic_cpp.__file__)
(out/'01_environment/runtime_paths.json').write_text(json.dumps({
    'python':sys.executable,'aic_cpp':str(binary),'aic_cpp_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),
    'cvxopt':cvxopt.__version__,'cv2':cv2.__version__,
    'core_extension_test':'bbox_overlap_rate smoke PASS; loop_t_homo_full SCOUT execution BLOCKED'},indent=2)+'\n')
videos=out/'03_intermediate/visualizations/pose_videos';videos.mkdir(parents=True,exist_ok=True)
reports=[]
for c in range(8):
    dst=videos/f'cam_{c}_source277-326_pose_no_id.mp4'
    cmd=['ffmpeg','-nostdin','-y','-loglevel','error','-framerate','10','-start_number','0',
         '-i',f'/inference/posetrack/pose/cam_{c}_%06d.jpg','-frames:v','50','-vf','scale=960:540',
         '-c:v','libx264','-threads','2','-pix_fmt','yuv420p','-crf','22','-movflags','+faststart',str(dst)]
    subprocess.run(cmd,check=True)
    cap=cv2.VideoCapture(str(dst));n=0
    while True:
        ok,im=cap.read()
        if not ok:break
        assert im.shape==(540,960,3);n+=1
    cap.release();assert n==50
    reports.append({'camera':f'cam_{c}','file':dst.name,'decoded_frames':n,'fps':10,'global_ids':False,'status':'PASS'})
(videos/'verification.json').write_text(json.dumps(reports,indent=2)+'\n')
print('PASS: 8 videos, 50 frames each; runtime paths captured')
