"""Original HCP grid and selection logic without the research coordinator."""
import hashlib, itertools, json
import _bootstrap
import numpy as np
import pandas as pd
import yaml
RUN = "hcp181_boundary_b122"


def points(m):
    g=m['grid'];keys=sorted(g)
    return [dict(zip(keys,p)) for p in itertools.product(*(g[k] for k in keys))]

def identity(method,fold,params):
    return (method,int(fold),json.dumps(params,sort_keys=True))

def task(kind,m,f,p,inner=-1):
    d=dict(kind=kind,method=m,outer_fold=int(f),params=p,inner_fold=int(inner))
    d['id']=kind+'_'+m+'_f'+str(f)+'_'+hashlib.sha1(json.dumps(d,sort_keys=True).encode()).hexdigest()[:12]
    return d

def build_initial(primary):
    cfg=yaml.safe_load((primary/'config.yaml').read_text())
    for m in cfg['experiment']['methods']:
        if m['name'] in ('graphnet','graph_tv'):m['grid']['lambda_g']=list(m['grid']['lambda_g'])+[.001,.003]
        if m['name']=='graph_tv':m['grid']['lambda_sel']=list(m['grid']['lambda_sel'])+[.00125]
    cfg['experiment']['name']=RUN
    cfg['experiment']['n_jobs']=1
    return cfg

def old_rows(cfg, primary):
    df=pd.read_csv(primary/'inner_scores.csv').fillna({'error':''})
    records={}
    for m in cfg['experiment']['methods']:
        for _,row in df[df.method.eq(m['name'])].iterrows():
            p={k:float(row[k]) for k in m['grid']}
            key=identity(m['name'],row.outer_fold,p)
            records.setdefault(key,[]).append(row.to_dict())
    for k,rows in records.items():
        assert len(rows)==3 and not any(r['error'] for r in rows),(k,len(rows))
    return records

def make_plan(primary):
    cfg=build_initial(primary);old=old_rows(cfg, primary);tasks=[]
    for m in cfg['experiment']['methods']:
        for f in range(5):
            for p in points(m):
                if identity(m['name'],f,p) not in old:
                    tasks.extend(task('inner',m['name'],f,p,i) for i in range(3))
    assert len(tasks)==525
    return cfg,tasks

def statistics(cfg,rows):
    from mlrdegg.experiment.cv import _pooled
    states={}
    for m in cfg['experiment']['methods']:
        for f in range(5):
            group=[]
            for p in points(m):
                key=identity(m['name'],f,p);r=rows[key]
                assert len(r)==3 and not any(x.get('error') for x in r)
                acc,loss,se=_pooled(r)
                repro=next((float(x['repro']) for x in r if pd.notna(x.get('repro',np.nan))),None)
                group.append(dict(key=key,params=p,accuracy=acc,log_loss=loss,score=-float(np.mean([x['log_loss'] for x in r])),
                    sample_se=se,fold_se=float(np.std([x['log_loss'] for x in r],ddof=1)/np.sqrt(3)),repro=repro))
            states[(m['name'],f)]=group
    return states

def needed_halves(states):
    tasks=[]
    for (m,f),ss in states.items():
        best=min(ss,key=lambda s:s['log_loss']);bound=best['log_loss']+max(best['sample_se'],best['fold_se'])
        minpoint=max(ss,key=lambda s:s['score'])
        for s in ss:
            if (s['log_loss']<=bound or s is minpoint) and s['repro'] is None:
                tasks.append(task('half',m,f,s['params']))
    return tasks

def select(states,half_results):
    repro={identity(r['task']['method'],r['task']['outer_fold'],r['task']['params']):r['metrics']['repro'] for r in half_results}
    chosen=[]
    for (m,f),ss in states.items():
        for s in ss:
            if s['repro'] is None:s['repro']=repro.get(s['key'])
        best=min(ss,key=lambda s:s['log_loss'])
        for rule in ('min','1se_repro','1se_fold_repro'):
            if rule=='min':win=max(ss,key=lambda s:s['score'])
            else:
                se=best['fold_se' if rule=='1se_fold_repro' else 'sample_se']
                eligible=[s for s in ss if s['log_loss']<=best['log_loss']+se]
                assert all(s['repro'] is not None for s in eligible)
                win=max(eligible,key=lambda s:(s['repro'],-s['log_loss']))
            chosen.append(dict(method=m,outer_fold=f,rule=rule,**{k:v for k,v in win.items() if k!='key'}))
    return chosen

def plan_refits(cfg,selection, primary):
    old=pd.read_csv(primary/'outer_results.csv')
    manifest=[];pending={}
    for s in selection:
        m=s['method'];f=s['outer_fold'];p=s['params']
        candidates=old[old.method.str.split('@').str[0].eq(m)&old.outer_fold.eq(f)]
        for k,v in p.items():candidates=candidates[np.isclose(candidates[k],v)]
        if len(candidates):
            row=candidates.iloc[0];source=dict(kind='active_primary',method=str(row.method),outer_fold=f,
                precision_source=str(row.get('reported_solution_source','existing')))
        else:
            t=task('refit',m,f,p);pending[t['id']]=t;source=dict(kind='new_refit',task_id=t['id'])
        manifest.append(dict(selection=s,source=source))
    return list(pending.values()),manifest
