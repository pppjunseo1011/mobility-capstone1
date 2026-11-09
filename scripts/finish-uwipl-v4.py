#!/usr/bin/env python3
"""Finish analysis/report only after all V4 inference jobs complete."""
import json,os,subprocess,sys,time
from pathlib import Path
R=Path(__file__).resolve().parents[1];O=R/'outputs/uwipl-v4-20261108'
for _ in range(120):
 state=json.loads((O/'run_state.json').read_text())
 if state['status']=='COMPLETE':break
 if state['status']=='FAILED':raise SystemExit(state)
 time.sleep(5)
else:raise SystemExit('Inference still running; no final report produced')
def cpu(script):
 subprocess.run(['docker','run','--rm','--name','scout-v4-report-'+script.split('-')[0],'--network','none','--cpus','3','--memory','8g','--user',f'{os.getuid()}:{os.getgid()}','-e','HOME=/tmp','-e','MPLCONFIGDIR=/tmp/matplotlib','-v',str(R)+':/project:ro','-v',str(R/'vendor')+':/vendor:ro','-v',str(O)+':/out','scout-mtmc/uwipl-botsort:8e23a69','python','/project/scripts/'+script],check=True)
cpu('analyze-uwipl-v4-losses.py')
subprocess.run([sys.executable,str(R/'scripts/report-uwipl-v4.py'),'collect'],check=True)
cpu('plot-uwipl-v4.py')
subprocess.run([sys.executable,str(R/'scripts/report-uwipl-v4.py'),'build'],check=True)
title='UWIPL_SCOUT_V4_검출진단_20261108';pdf=O/'report'/(title+'.pdf');html=O/'report'/(title+'.html')
subprocess.run(['google-chrome','--headless','--no-sandbox','--disable-gpu','--disable-background-networking','--no-first-run','--user-data-dir=/tmp/scout-v4-pdf-profile','--no-pdf-header-footer','--print-to-pdf='+str(pdf),html.as_uri()],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,timeout=90)
print('PDF_READY',pdf,flush=True)
