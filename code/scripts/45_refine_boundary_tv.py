"""B122-R07: warm-state precision work for five fixed selected TV fits only."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse,dataclasses,json,sys,time,traceback,subprocess
from pathlib import Path

RUN='hcp181_boundary_b122'

def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x)),encoding='utf-8')
    for attempt in range(20):
        try:os.replace(tmp,path);return
        except PermissionError:
            if attempt==19:raise
            time.sleep(.2)

def save(path,value):
    import joblib
    tmp=path.with_suffix('.tmp');joblib.dump(value,tmp)
    for attempt in range(20):
        try:os.replace(tmp,path);return
        except PermissionError:
            if attempt==19:raise
            time.sleep(.2)

def worker(root,ident):
    run=root/'results'/RUN;pr=run/'precision';out=pr/'tasks'/ident;out.mkdir(parents=True,exist_ok=True)
    import msvcrt
    lock=(out/'worker.lock').open('a+b');lock.seek(0)
    try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    except OSError:return 2  # never create a duplicate fit
    completion=out/'completion.json'
    if completion.exists():return 0 if json.loads(completion.read_text(encoding='utf-8'))['state']=='completed' else 1
    sys.path[:0]=[str(run/'runtime/code/src'),str(run/'runtime/code/scripts')]
    import numpy as np,pandas as pd,joblib
    import _bootstrap
    from mlrdegg.experiment.cv import outer_splits
    from mlrdegg.experiment.models import fit_info
    from mlrdegg.experiment.metrics import classification_metrics
    from mlrdegg.solver import SolverState,solve_mm_admm,objective
    from mlrdegg.residual import common_residual
    src=run/'tasks'/ident;started=time.monotonic();spent=0.;latest={};diag={}
    def elapsed():return spent+time.monotonic()-started
    def status(**fields):atomic(out/'status.json',{'id':ident,'state':'running','pid':os.getpid(),'elapsed_seconds':elapsed(),**fields})
    class BandPassed(Exception):pass
    try:
        status(phase='loading')
        old=json.loads((src/'summary.json').read_text(encoding='utf-8'));task=old['task']
        assert task['method']=='graph_tv' and task['params']=={'lambda_g':.01,'lambda_sel':.00125}
        pipe=joblib.load(src/'pipeline.pkl');clf=pipe.named_steps['clf'];original=clf.fit_log_[-1][1]
        assert np.array_equal(original.Theta,np.vstack([clf.B_,clf.c_[None,:]]))
        data=joblib.load(run/'prepared/metadata.pkl');X=np.load(run/'prepared/X.npy',mmap_mode='r')
        cfg=_bootstrap.load_config(run/'inputs/config.yaml');tr,te=outer_splits(cfg['experiment'],data['y'],data['groups'])[task['outer_fold']]
        oldpred=pd.read_csv(src/'predictions.csv').sort_values('sample')
        assert np.array_equal(te,oldpred['sample']) and np.array_equal(data['y'][te],oldpred.y_true)
        Xtr=pipe.named_steps['scale'].transform(X[tr]);Xa=np.column_stack([Xtr,np.ones(len(tr))]);del Xtr
        yi=clf.label_encoder_.transform(data['y'][tr]);Y=np.eye(len(clf.classes_))[yi];H=clf.H_;branches=clf.branches_
        prior=np.bincount(yi,minlength=len(clf.classes_))/len(yi);logp=np.log(np.maximum(prior,1e-12))
        null=np.zeros_like(original.Theta);null[-1]=H.T@(logp-logp.mean())
        assert np.isclose(objective(Xa,Y,H,original.Theta,branches,clf.epsilon),old['metrics']['objective'],atol=1e-10,rtol=0)
        initial=SolverState(original.Theta,original.U,original.Z,original.rho)
        opts=dataclasses.replace(clf._options(),max_outer=5000,max_inner=1000,tol_obj=1e-8,tol_x=1e-5,
                                cg_tol=1e-8,inner_rel_tol0=1e-5,inner_rel_tol_min=1e-8,
                                adapt_rho_outer=0,resume_history=None,resume_converged=False,
                                resume_rho_updates=0,resume_kkt_violations=0)
        checkpoint=out/'precision_checkpoint.pkl'
        if checkpoint.exists():
            latest=joblib.load(checkpoint);initial=latest['state'];spent=latest['elapsed_seconds']
            opts=dataclasses.replace(opts,resume_history=latest['history'],resume_converged=False,
                                    resume_rho_updates=latest['rho_updates'],resume_kkt_violations=latest['kkt_violations'])
        atomic(out/'provenance.json',dict(task=task,source=str(src/'pipeline.pkl'),source_iteration=original.n_outer,
              objective_match=True,model_and_scaler_unchanged=True,extra_budget_hours=4,diagnostic_budget_seconds=120,
              target='unchanged common_residual_1e-03 pass (estimate <=1e-3 and error bound <=1e-5)',
              strict_options={k:getattr(opts,k) for k in ['tol_obj','tol_x','max_outer','max_inner','cg_tol','inner_rel_tol0','inner_rel_tol_min']}))
        last_save=[0.];last_diag=[0.]
        def monitor(theta,row):
            nonlocal latest,diag
            frame=sys._getframe(1).f_locals
            stopped=abs(frame['dF'])<=opts.tol_obj*max(1.,abs(frame['F'])) and frame['step']<=opts.tol_x*max(np.linalg.norm(theta),1e-12)
            check=len(frame['history'])%25==0 or stopped
            expired=elapsed()>=4*3600
            if time.monotonic()-last_save[0]>=120 or check or expired:
                latest=dict(state=SolverState(theta.copy(),[u.copy() for u in frame['U']],[z.copy() for z in frame['Z']],float(frame['rho'])),
                            history=list(frame['history']),rho_updates=frame['n_rho_updates'],kkt_violations=frame['n_kkt_violations'],
                            objective=float(frame['F']),elapsed_seconds=elapsed())
                save(checkpoint,latest);status(phase='refining',iteration=len(latest['history']),objective=latest['objective'],latest_diagnostic=diag)
                last_save[0]=time.monotonic()
            if check and len(frame['history'])!=last_diag[0]:
                status(phase='precision_diagnostic',iteration=len(frame['history']),latest_diagnostic=diag)
                diag=common_residual(Xa,Y,H,branches,theta,null,clf.epsilon,budget=120)
                last_diag[0]=len(frame['history'])
                atomic(out/f'diagnostic_{len(frame["history"]):05d}.json',diag)
                status(phase='refining',iteration=len(frame['history']),objective=float(frame['F']),latest_diagnostic=diag)
                if diag.get('common_residual_1e-03')=='pass':raise BandPassed()
            if expired:raise TimeoutError('Four additional hours exhausted; checkpoint retained')
        opts.monitor=monitor
        try:
            refined=solve_mm_admm(Xa,Y,H,branches,opts,initial)
            diag=common_residual(Xa,Y,H,branches,refined.Theta,null,clf.epsilon,budget=120)
        except BandPassed:
            st=latest['state'];hist=latest['history']
            refined=dataclasses.replace(original,Theta=st.Theta,U=st.U,Z=st.Z,rho=st.rho,objective=latest['objective'],
                    n_outer=len(hist),n_inner_total=sum(x['inner_iterations'] for x in hist),converged=False,status='precision_band_passed',
                    history=hist,runtime=elapsed(),n_kkt_violations=latest['kkt_violations'],diagnostics={'precision_band_stopped':True,'iteration_limit_hit':False})
        save(out/'refined_result.pkl',refined)
        if diag.get('common_residual_1e-03')!='pass':
            atomic(out/'final_diagnostic.json',diag);raise RuntimeError('Strict solver ended without passing original common-residual band')
        clf.B_=refined.Theta[:-1];clf.c_=refined.Theta[-1];clf.coef_=clf.B_@H.T;clf.intercept_=H@clf.c_
        clf.split_={b.name:u for b,u in zip(branches,refined.U)};clf.objective_=refined.objective
        clf.converged_=refined.converged;clf.status_=refined.status;clf.n_outer_=refined.n_outer;clf.n_inner_=refined.n_inner_total
        clf.n_kkt_violations_=refined.n_kkt_violations;clf.history_=refined.history;clf.runtime_=elapsed()
        clf.diagnostics_={**refined.diagnostics,**diag};clf.common_residual_=diag;clf.fit_log_.append(('precision',refined))
        proba=pipe.predict_proba(X[te]);pred=clf.classes_[proba.argmax(axis=1)];losses=-np.log(np.clip(proba[np.arange(len(te)),np.searchsorted(clf.classes_,data['y'][te])],1e-15,None))
        metrics={**classification_metrics(data['y'][te],proba,clf.classes_),**fit_info(pipe),'n_test':len(te),'log_loss_sd':float(losses.std(ddof=1)),
                 'fit_time':old['metrics']['fit_time']+elapsed(),'precision_fit_time':elapsed(),'source_original_n_outer':original.n_outer,'error':''}
        oldpred['y_pred']=pred;oldpred.to_csv(out/'predictions.csv',index=False)
        importance=clf.feature_importance()
        np.savez_compressed(out/'importance.npz',importance=importance);save(out/'pipeline.pkl',pipe)
        pd.DataFrame({'subject':data['groups']['subject'][te],'loss':losses}).groupby('subject').agg(loss=('loss','mean'),n=('loss','size')).to_csv(out/'participant_losses0.csv')
        atomic(out/'summary.json',dict(task=task,metrics=metrics,source='B122-R07 fixed-selected warm-state precision refinement'))
        atomic(completion,dict(state='completed',elapsed_seconds=elapsed(),precision_band='pass',strict_solver_converged=refined.converged))
        status(state='completed',phase='completed',latest_diagnostic=diag)
        return 0
    except Exception:
        atomic(completion,dict(state='attention',elapsed_seconds=elapsed(),error=traceback.format_exc(),resumable=True))
        status(state='attention',phase='stopped',latest_diagnostic=diag);traceback.print_exc();return 1

def driver(root,host):
    pr=root/'results'/RUN/'precision';stage=pr/host;stage.mkdir(parents=True,exist_ok=True)
    done=stage/'completion.json'
    if done.exists():return
    atomic(stage/'status.json',dict(host=host,state='preparing',pid=os.getpid()))
    tasks=json.loads((pr/'manifest.json').read_text(encoding='utf-8'))['tasks'];tasks=[t for t in tasks if t['host']==host]
    running=[]
    for t in tasks:
        out=pr/'tasks'/t['id'];out.mkdir(parents=True,exist_ok=True);log=(out/'worker.log').open('a',encoding='utf-8')
        proc=subprocess.Popen([sys.executable,__file__,'--root',str(root),'--worker',t['id']],stdout=log,stderr=log,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        running.append((t,proc,log));time.sleep(2)
    while True:
        rows=[]
        for task,proc,log in running:
            out=pr/'tasks'/task['id'];row={'id':task['id'],'pid':proc.pid,'process_alive':proc.poll() is None}
            for name in ['status','completion']:
                p=out/(name+'.json')
                if p.exists():
                    try:row[name]=json.loads(p.read_text(encoding='utf-8'))
                    except (OSError,ValueError):pass
            rows.append(row)
        atomic(stage/'status.json',dict(host=host,state='running',pid=os.getpid(),active=sum(r['process_alive'] for r in rows),tasks=rows))
        if not any(r['process_alive'] for r in rows):
            for _,_,log in running:log.close()
            atomic(done,dict(host=host,state='completed' if all(r.get('completion',{}).get('state')=='completed' for r in rows) else 'attention',tasks=rows));return
        time.sleep(20)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--host');p.add_argument('--worker');a=p.parse_args()
    if a.worker:sys.exit(worker(a.root,a.worker))
    try:driver(a.root,a.host)
    except Exception:
        atomic(a.root/'results'/RUN/'precision'/a.host/'driver_failure.json',dict(error=traceback.format_exc()))
        raise
