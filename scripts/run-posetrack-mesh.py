#!/usr/bin/env python3
"""Run full-field B/C from cached inputs, with versioned native-source patch."""
import argparse,difflib,hashlib,importlib.util,json,sys
from pathlib import Path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['B_no_ground','C_mesh'],required=True)
    ap.add_argument('--input',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--frames',type=int,default=50);ap.add_argument('--solver',choices=['glpk','highs'],default='glpk')
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,'/opt/posetrack/track')
    import Tracker.PoseTracker as M
    original=Path(M.__file__).read_text();host=Path('/project/repos/posetrack/track/Tracker/PoseTracker.py').read_text();assert original==host
    code=original
    # NumPy rows are views: tuple assignment otherwise overwrites the second
    # source before it is copied during an occlusion-driven view exchange.
    for field in ['age_2D','keypoints_mv','bbox_mv']:
        old=f'self.{field}[v], track.{field}[v] = track.{field}[v], self.{field}[v]'
        assert code.count(old)==1
        code=code.replace(old,f'self.{field}[v], track.{field}[v] = track.{field}[v].copy(), self.{field}[v].copy()')
    start=code.index('    def get_output(self):');end=code.index('    def single_view_init(',start)
    code=code[:start]+'''    def get_output(self):
        return self.cameras[0].scout_geometry.output(self)

'''+code[end:]
    start=code.index('    def compute_epi_homo_aff(');end=code.index('    def compute_bboxiou_aff(',start)
    code=code[:start]+'''    def compute_epi_homo_aff(self, detection_sample_list_mv, avail_tracks):
        return self.cameras[0].scout_geometry.affinity(self, detection_sample_list_mv, avail_tracks)

'''+code[end:]
    start=code.index('        mv_rays = self.CalcJointRays',code.index('    def target_init('));end=code.index('        aff_final = 2*aff_epi + aff_homo',start)
    code=code[:start]+'''        aff_epi, aff_homo = self.cameras[0].scout_geometry.init_affinities(self, detection_sample_list_mv)

'''+code[end:]
    old='if (joint_3d[2]<-1 or joint_3d[2]>2.5) or (j_idx in self.feet_idx and (joint_3d[2]<-1 or joint_3d[2]>1)):'
    assert code.count(old)==1
    code=code.replace(old,'if self.cameras[0].scout_geometry.invalid_height(self, joint_3d, j_idx):')
    if args.mode=='B_no_ground':code=code.replace('        a_homo = 1','        a_homo = 0')
    patchpath=args.out/'PoseTracker.patched.py';patchpath.write_text(code)
    (args.out/'native_changes.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),code.splitlines(True),fromfile='original/PoseTracker.py',tofile=args.mode+'/PoseTracker.py')))
    M.__file__=str(patchpath);exec(compile(code,str(patchpath),'exec'),M.__dict__)
    if args.solver=='highs':
        import numpy as np,time
        from scipy.optimize import milp,Bounds,LinearConstraint
        from scipy.sparse import coo_matrix
        def exact_highs(coeff, sparse, upper):
            v,ri,ci=sparse;n=len(coeff);A=coo_matrix((v,(ri,ci)),shape=(len(upper),n)).tocsc()
            A.indices=A.indices.astype(np.int32);A.indptr=A.indptr.astype(np.int32)
            start=time.monotonic()
            result=milp(-coeff,integrality=np.ones(n),bounds=Bounds(np.zeros(n),np.ones(n)),
              constraints=LinearConstraint(A,-np.inf,np.asarray(upper)),options={'time_limit':180,'mip_rel_gap':0,'presolve':True})
            record={'variables':n,'constraints':len(upper),'status':int(result.status),'message':result.message,
              'objective':None if result.fun is None else float(result.fun),'elapsed_seconds':time.monotonic()-start,
              'mip_gap':getattr(result,'mip_gap',None)}
            with (args.out/'solver.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            if result.status!=0 or result.x is None:raise RuntimeError('Exact BIP solver not optimal: '+result.message)
            x=np.rint(result.x).astype(int)
            assert np.max(A@x-np.asarray(upper))<1e-6
            return x
        M.GLPKSolver._solve_bip=staticmethod(exact_highs)
    spec=importlib.util.spec_from_file_location('scout_mesh_geometry','/project/scripts/posetrack-mesh-geometry.py');G=importlib.util.module_from_spec(spec);sys.modules[spec.name]=G;spec.loader.exec_module(G)
    source=Path('/project/scripts/trace-posetrack-flat.py').read_text()
    source=source.replace("O=Path('/out');FRAME",f"O=Path({str(args.out)!r});I=Path({str(args.input)!r});FRAME")
    begin=source.index('ORIG_GEO=');end=source.index('ORIG_ASSIGN=',begin)
    source=source[:begin]+'''ORIG_GEO=M.PoseTracker.compute_epi_homo_aff
def geometry(self,dets,tracks):
    result=ORIG_GEO(self,dets,tracks)
    self._audit_geo=([a.copy() for a in result[0]],[a.copy() for a in result[1]])
    assert all(np.isfinite(a).all() for parts in [result] for arrs in parts for a in arrs)
    return result
M.PoseTracker.compute_epi_homo_aff=geometry

'''+source[end:]
    source=source.replace("O/f'calibration/", "I/f'calibration/").replace("(O/'observations.json')", "(I/'observations.json')").replace("O/f'input/", "I/f'input/")
    source=source.replace('tracker=M.PoseTracker(cams);start=', f"geo=G.install(M,cams,'/data',obs,{args.mode!r});tracker=M.PoseTracker(cams);start=")
    source=source.replace('range(50):',f'range({args.frames}):').replace("'frames':50",f"'frames':{args.frames}")
    source=source.replace('      detections=[]','      geo.frame=FRAME\n      detections=[]')
    source=source.replace("assert image_source==host_source", "assert hashlib.sha256(host_source).hexdigest()==ORIGINAL_SHA")
    source=source.replace("'image_and_host_tracker_equal':True", "'image_and_host_tracker_equal':False,'original_sha256':ORIGINAL_SHA,'geometry_mode':MODE")
    source=source.replace("'tracking_source_modified':False", "'tracking_source_modified':True,'geometry_mode':MODE")
    source=source.replace("'geometry_cpp_equivalence':'PASS all observed pairs, atol=rtol=1e-7'", "'geometry_cpp_equivalence':'see separate geometry verification; C uses modified mesh term'")
    source=source.replace("'bbox_space':'undistorted'", "'bbox_space':'undistorted','detection_confidence':float(t.bbox_mv[cam,4]),'local_id':None")
    source=source.replace("L=inspect.currentframe().f_back.f_locals;r=OLD_BIP(self,aff,rtn_matrix)","L=inspect.currentframe().f_back.f_locals\n    np.save(O/f'initialization_affinity_frame_{FRAME:03d}.npy',aff)\n    print('BIP_BEGIN',FRAME,len(aff),flush=True)\n    r=OLD_BIP(self,aff,rtn_matrix)")
    source=source.replace("'norm':L['norm'][i,j]", "'norm':L['norm'][i,j],'a_box':L['a_box'],'a_epi':L['a_epi'],'a_homo':L['a_homo'],'a_reid':L['a_reid']")
    source=source.replace("    print('DONE',", "    js(O/'mesh_events.json',geo.events);js(O/'mesh_counts.json',dict(geo.counts))\n    print('DONE',")
    (args.out/'trace_runner.executed.py').write_text(source)
    (args.out/'configuration.json').write_text(json.dumps({'mode':args.mode,'frames':args.frames,'source_start':277,'input':str(args.input),
      'roi':None,'epi_threshold':.2,'ground_threshold':1.5,'ankle_offset_world_z':.15,'fps_source':10,
      'original_frame_age_thresholds_retained':True,'height_reference':'mesh vertical ground, both B and C',
      'B_C_difference':'ground affinity disabled vs mesh affinity in track association AND target initialization',
      'output_world':'mesh projection or triangulation, both B and C','solver':args.solver,
      'common_bugfix':'switch_view NumPy row copies avoid aliasing; applies equally to B and C','gt_input':False},indent=2)+'\n')
    ns={'__name__':'__main__','G':G,'ORIGINAL_SHA':hashlib.sha256(original.encode()).hexdigest(),'MODE':args.mode}
    try:exec(compile(source,str(args.out/'trace_runner.executed.py'),'exec'),ns)
    except BaseException:
        import traceback
        (args.out/'failure.txt').write_text(traceback.format_exc())
        for key in ['EVENTS','PAIRS','OUTPUT','STATELOG']:
            if key in ns and 'js' in ns:ns['js'](args.out/(key.lower()+'.partial.json'),ns[key])
        raise
if __name__=='__main__':main()
