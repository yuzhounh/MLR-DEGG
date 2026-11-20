"""B122 X01/X02: frozen-grid extension and inner-fold one-SE sensitivity.

Worker entry point. Existing results are read-only. Each fit is a subprocess with
one BLAS thread, an explicit timeout, and atomic completed-fit checkpoints.
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import dataclasses
import importlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

RUN = 'hcp181_boundary_b122'

def atomic(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    os.replace(tmp,path)

def boot(run):
    sys.path[:0] = [str(run/'runtime/src'),str(run/'runtime/scripts')]

def save_pickle(path, value):
    import joblib
    tmp = path.with_suffix('.tmp'); joblib.dump(value,tmp); os.replace(tmp,path)

def worker(root, task_file):
    import numpy as np
    import pandas as pd
    import joblib
    run = root/'results'/RUN; boot(run)
    import _bootstrap
    from mlrdegg.experiment.cv import (outer_splits, split, split_halves, coef_map)
    from mlrdegg.experiment.models import Context, build_graphs, make_estimator, fit_info, importance_map
    from mlrdegg.experiment.metrics import classification_metrics
    from mlrdegg.solver import SolverState, solve_mm_admm
    task = json.loads(task_file.read_text(encoding='utf-8'))
    out = run/'tasks'/task['id']; out.mkdir(parents=True,exist_ok=True)
    if (out/'completion.json').exists():
        return 0 if json.loads((out/'completion.json').read_text())['state']=='completed' else 1
    started = time.monotonic()
    def status(**kw):
        atomic(out/'status.json',dict(id=task['id'],pid=os.getpid(),state='running',
               elapsed_seconds=time.monotonic()-started,**kw))
    try:
        cfg = _bootstrap.load_config(run/'inputs/config.yaml'); exp=cfg['experiment']
        method=next(m for m in exp['methods'] if m['name']==task['method'])
        params=task['params']; fold=task['outer_fold']; kind=task['kind']
        X=np.load(run/'prepared/X.npy',mmap_mode='r')
        data=joblib.load(run/'prepared/metadata.pkl'); y=data['y']; groups=data['groups']
        spatial,temporal=build_graphs(cfg['graph'],data['space'],data['n_time'])
        ctx=Context(data['n_space'],data['n_time'],spatial,temporal)
        tr,te=outer_splits(exp,y,groups)[fold]
        if kind=='inner':
            itr,iva=split(exp['inner_cv'],y[tr],{k:v[tr] for k,v in groups.items()})[task['inner_fold']]
            pairs=[(tr[itr],tr[iva])]
        elif kind=='half':
            halves=split_halves(tr,y,exp['split_half_seed']+fold,groups[exp['split_half_group']])
            pairs=[(h,None) for h in halves]
        else: pairs=[(tr,te)]
        results=[]
        for index,(train,test) in enumerate(pairs):
            result_path=out/f'fit{index}.pkl'
            if result_path.exists():
                results.append(joblib.load(result_path)); continue
            pipe=make_estimator(method,params,ctx,exp['solver']); clf=pipe.named_steps['clf']
            status(phase='fitting',half=index)
            # Same model, objective and stopping criteria as the frozen analysis.
            # Optional MM checkpoints preserve accepted iterations without changing updates.
            original_solve=clf._solve
            def solve(Xa,Y,H,branches,opts,initial):
                from mlrdegg.proxgrad import apg_applicable
                if clf.solver=='apg' or (clf.solver=='auto' and apg_applicable(branches)):
                    return original_solve(Xa,Y,H,branches,opts,initial)
                checkpoint=out/f'mm{index}.pkl'; last=[0.0]
                if checkpoint.exists():
                    saved=joblib.load(checkpoint)
                    initial=saved['state']
                    opts=dataclasses.replace(opts,resume_history=saved['history'],
                        resume_converged=saved['converged'],resume_rho_updates=saved['rho_updates'],
                        resume_kkt_violations=saved['kkt_violations'])
                def monitor(theta,row):
                    frame=sys._getframe(1).f_locals
                    if time.monotonic()-last[0]>=120:
                        stopped=(abs(frame['dF'])<=opts.tol_obj*max(1.,abs(frame['F'])) and
                                 frame['step']<=opts.tol_x*max(np.linalg.norm(theta),1e-12))
                        save_pickle(checkpoint,dict(state=SolverState(theta.copy(),
                            [a.copy() for a in frame['U']],[a.copy() for a in frame['Z']],float(frame['rho'])),
                            history=list(frame['history']),converged=bool(stopped),
                            rho_updates=frame['n_rho_updates'],kkt_violations=frame['n_kkt_violations']))
                        status(phase='fitting',half=index,iteration=int(row['outer']),objective=row['objective'])
                        last[0]=time.monotonic()
                opts.monitor=monitor
                return original_solve(Xa,Y,H,branches,opts,initial)
            clf._solve=solve
            pipe.fit(X[train],y[train]); del clf.__dict__['_solve']
            info=fit_info(pipe)
            if kind=='half': result={'info':info,'coef':coef_map(pipe)}
            else:
                proba=pipe.predict_proba(X[test]); classes=pipe.classes_
                losses=-np.log(np.clip(proba[np.arange(len(test)),np.searchsorted(classes,y[test])],1e-15,None))
                scores={**classification_metrics(y[test],proba,classes),**info,
                        'n_test':len(test),'log_loss_sd':float(losses.std(ddof=1)),
                        'fit_time':time.monotonic()-started,'error':''}
                losses_df=pd.DataFrame({'subject':groups['subject'][test],'loss':losses})
                losses_df.groupby('subject',sort=True).agg(loss=('loss','mean'),n=('loss','size')).to_csv(out/f'participant_losses{index}.csv')
                result={'info':scores,'importance':importance_map(pipe,ctx) if kind=='refit' else None}
                if kind=='refit':
                    pd.DataFrame({'sample':test,'subject':groups['subject'][test],
                        'y_true':y[test],'y_pred':classes[proba.argmax(axis=1)]}).to_csv(out/'predictions.csv',index=False)
                    save_pickle(out/'pipeline.pkl',pipe)
            save_pickle(result_path,result);results.append(result)
        if kind=='half':
            a,b=[r['coef'] for r in results]
            if a.std()==0 or b.std()==0: repro=0.0
            else: repro=float(np.corrcoef(a,b)[0,1])
            summary={'repro':repro,'half0':results[0]['info'],'half1':results[1]['info']}
        else: summary=results[0]['info']
        atomic(out/'summary.json',dict(task=task,metrics=summary))
        atomic(out/'completion.json',dict(state='completed',elapsed_seconds=time.monotonic()-started))
        return 0
    except Exception:
        atomic(out/'completion.json',dict(state='failed',error=traceback.format_exc(),
            elapsed_seconds=time.monotonic()-started))
        traceback.print_exc();return 1

def prepare(root):
    import numpy as np
    import joblib
    run=root/'results'/RUN;boot(run)
    if (run/'prepared/metadata.pkl').exists() and (run/'prepared/X.npy').exists():return
    import _bootstrap
    from mlrdegg.datasets import load_derived
    cfg=_bootstrap.load_config(run/'inputs/config.yaml')
    ds=load_derived(root/cfg['preprocess']['out_dir'],cfg['experiment']['subjects'])
    _,X,y,groups=next(importlib.import_module('03_run_decoding').make_units(ds,'pooled'))
    assert len(ds.subjects)==181 and X.shape==(11584,19204)
    (run/'prepared').mkdir(exist_ok=True)
    np.save(run/'prepared/X.tmp.npy',X);os.replace(run/'prepared/X.tmp.npy',run/'prepared/X.npy')
    save_pickle(run/'prepared/metadata.pkl',dict(y=y,groups=groups,space=ds.space,
                n_space=ds.n_space,n_time=ds.n_time))

def free_gib():
    import ctypes
    class Memory(ctypes.Structure):
        _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong),
          ('total',ctypes.c_ulonglong),('avail',ctypes.c_ulonglong),
          ('page_total',ctypes.c_ulonglong),('page_avail',ctypes.c_ulonglong),
          ('virtual_total',ctypes.c_ulonglong),('virtual_avail',ctypes.c_ulonglong),('extended',ctypes.c_ulonglong)]
    m=Memory();m.length=ctypes.sizeof(m);ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.avail/2**30

def stage(root, host, phase, slots):
    import joblib
    import numpy as np
    run=root/'results'/RUN
    stage_dir=run/phase;stage_dir.mkdir(exist_ok=True)
    done=stage_dir/'completion.json'
    if done.exists():return 0 if json.loads(done.read_text())['state']=='completed' else 1
    atomic(stage_dir/'status.json',dict(state='preparing',pid=os.getpid(),host=host,workers=slots))
    prepare(root)
    pending=json.loads((run/'plans'/f'{phase}_{host}.json').read_text(encoding='utf-8'))
    # Alternate heavy and light fits so both servers use twenty slots efficiently.
    heavy=[t for t in pending if t['method']=='graph_tv' and t['params'].get('lambda_g',0)>0]
    light=[t for t in pending if t not in heavy]
    pending=[]
    while heavy or light:
        if heavy:pending.append(heavy.pop(0))
        if light:pending.append(light.pop(0))
    running=[];failed=[];completed=[];max_active=0;started=time.monotonic();last=0
    while pending or running:
        if time.monotonic()-started>96*3600:
            failed.append({'reason':'96-hour stage wall-clock budget'});pending=[]
        for item in list(running):
            p,t,start,log=item
            timeout=(8 if t['kind']=='half' else 4)*3600
            if p.poll() is None and time.monotonic()-start>timeout:
                subprocess.run(['taskkill','/PID',str(p.pid),'/T','/F'],capture_output=True)
                atomic(run/'tasks'/t['id']/'completion.json',dict(state='timeout',seconds=timeout))
                p.wait()
            if p.poll() is not None:
                log.close();running.remove(item)
                (completed if p.returncode==0 else failed).append(t['id'])
        # Preserve failures; no automatic retries. Other independent tasks may finish.
        while pending and len(running)<slots:
            available=free_gib();candidate=None
            # Reserve 28 GiB and account for recently launched fits still allocating.
            for i,t in enumerate(pending):
                estimate=(21 if phase=='refit' else 13 if phase=='inner' else 10) if t['method']=='graph_tv' and t['params'].get('lambda_g',0)>0 else 5
                young=sum(x[2]>time.monotonic()-180 for x in running)
                if available>=28+estimate+young*5:candidate=i;break
            if candidate is None:break
            t=pending.pop(candidate);out=run/'tasks'/t['id'];out.mkdir(parents=True,exist_ok=True)
            tf=out/'task.json';atomic(tf,t)
            log=(out/'worker.log').open('a',encoding='utf-8')
            p=subprocess.Popen([sys.executable,__file__,'--root',str(root),'--worker',str(tf)],
                stdout=log,stderr=log,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            running.append((p,t,time.monotonic(),log));max_active=max(max_active,len(running))
            time.sleep(2)
        if time.monotonic()-last>30:
            atomic(stage_dir/'status.json',dict(state='running',host=host,phase=phase,pid=os.getpid(),
                configured_workers=slots,active=len(running),peak_active=max_active,completed=len(completed),
                failed=len(failed),pending=len(pending),free_gib=free_gib(),elapsed_seconds=time.monotonic()-started,
                active_tasks=[{'id':t['id'],'pid':p.pid,'seconds':time.monotonic()-st} for p,t,st,_ in running]))
            last=time.monotonic()
        time.sleep(3)
    summaries=[]
    for task_id in completed:
        p=run/'tasks'/task_id/'summary.json'
        if p.exists(): summaries.append(json.loads(p.read_text(encoding='utf-8')))
    atomic(stage_dir/'summaries.json',summaries)
    # Export only small results; fitted models and solver checkpoints stay remote.
    if phase=='refit':
        import zipfile
        with zipfile.ZipFile(stage_dir/'refit_outputs.zip','w',zipfile.ZIP_DEFLATED) as z:
            for task_id in completed:
                p=run/'tasks'/task_id
                for name in ('summary.json','predictions.csv'):
                    if (p/name).exists():z.write(p/name,f'{task_id}/{name}')
                fit=p/'fit0.pkl'
                if fit.exists():
                    im=joblib.load(fit)['importance'];np.savez_compressed(p/'importance.npz',importance=im)
                    z.write(p/'importance.npz',f'{task_id}/importance.npz')
    atomic(done,dict(state='failed' if failed else 'completed',host=host,phase=phase,
        completed=len(completed),failed=failed,peak_active=max_active,elapsed_seconds=time.monotonic()-started))
    return int(bool(failed))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--host');ap.add_argument('--phase',choices=['inner','half','refit'])
    ap.add_argument('--workers',type=int,default=1);ap.add_argument('--worker',type=Path)
    a=ap.parse_args()
    if a.worker:return worker(a.root,a.worker)
    assert 1<=a.workers<=38
    try:return stage(a.root,a.host,a.phase,a.workers)
    except Exception:
        atomic(a.root/'results'/RUN/a.phase/'completion.json',
               dict(state='failed',error=traceback.format_exc(),failed=['controller_error']))
        raise

if __name__=='__main__':sys.exit(main())
