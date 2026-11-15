"""Analyze retained B122 outputs only; preserve the original HCP comparisons."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import argparse,importlib,json
from pathlib import Path
import numpy as np,pandas as pd
import _bootstrap
ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'results/hcp181_boundary_b122';OUT=RUN/'analysis'
report=importlib.import_module('paper_report')

def render_table(sums,tests,out,install=True):
    """Pure formatting of saved statistics; keep the precision source explicit."""
    lines=[]
    for rule,heading in [('min','min'),('1se_repro','sample-level one-SE'),('1se_fold_repro','inner-fold-level one-SE')]:
        if lines:
            lines.append(r'\midrule')
        lines.append(r'\multicolumn{7}{l}{\emph{'+heading+r'} rule}\\')
        for method in ['group_lasso','graphnet','graph_tv','superquadratic']:
            r=sums[sums.rule.eq(rule)&sums.method.eq(method)].iloc[0]
            t=tests[tests.rule.eq(rule)&tests.method.eq(method)]
            delta='--' if t.empty else f"${100*t.iloc[0]['diff']:+.2f}$"
            pv='--' if t.empty else report.format_p(t.iloc[0].p_holm)
            lines.append(f"{report.LABEL[method]} & {100*r.accuracy:.2f} ({100*r.accuracy_sd:.2f}) & {r.log_loss:.3f} & {r.R:.3f} & {100*r.nonzero:.1f} & {delta} & {pv}"+r'\\')
    table='\\newcommand{\\BoundaryRows}{%\n'+'\n'.join(lines)+'\n}\n'
    if out is not None:
        (out/'table_rows.tex').write_text(table,encoding='utf-8')
    if install:
        (ROOT/'manuscript/tables/b122_boundary_rows.tex').write_text(table,encoding='utf-8')


def render_saved_table():
    if not (RUN/'precision/accepted.json').exists():
        raise RuntimeError('Accepted precision source required for the active S10 table')
    saved=RUN/'precision/analysis'
    render_table(pd.read_csv(saved/'summary.csv'),pd.read_csv(saved/'tests.csv'),None)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--original',action='store_true',help='Reproduce retained pre-precision analysis without replacing the active manuscript table')
    args=parser.parse_args()
    precision=(RUN/'precision/accepted.json').exists() and not args.original
    OUT=RUN/('precision/analysis' if precision else 'analysis')
    OUT.mkdir(exist_ok=True)
    base=ROOT/'results/hcp181_merged/primary'
    old=pd.read_csv(base/'outer_results.csv');pred=pd.read_csv(base/'units/pooled_predictions.csv')
    with np.load(base/'units/pooled_maps.npz') as z:oldmaps={k:z[k] for k in z.files}
    outer=[];predictions=[];maps={};sources=[];precision_ids=set();precision_rows=0
    manifest=json.loads((RUN/'refit_manifest.json').read_text())
    for item in manifest:
        s=item['selection'];src=item['source'];m=s['method'];fold=s['outer_fold'];rule=s['rule']
        name=m if rule=='min' else m+'@'+rule
        if src['kind']=='active_primary':
            row=old[old.method.eq(src['method'])&old.outer_fold.eq(fold)].iloc[0].to_dict()
            p=pred[pred.method.eq(src['method'])&pred.outer_fold.eq(fold)].copy()
            im=oldmaps[src['method']][fold]
        else:
            tid=src['task_id'];host=next(h for h in ['partition_a','partition_b'] if (RUN/f'collected/refit_{h}'/tid).exists())
            d=RUN/f'collected/refit_{host}'/tid
            if precision:
                candidate=RUN/f'precision/collected/{host}'/tid
                if candidate.exists():
                    assert m=='graph_tv'
                    d=candidate;precision_ids.add(tid);precision_rows+=1
                    item={**item,'precision_source':str(d.relative_to(ROOT))}
            row=json.loads((d/'summary.json').read_text())['metrics']
            if precision and tid in precision_ids:
                assert row['common_residual_1e-03']=='pass' and row['common_residual']<=1e-3 and row['common_residual_error_bound']<=1e-5
            p=pd.read_csv(d/'predictions.csv')
            with np.load(d/'importance.npz') as z:im=z['importance']
        p=p.sort_values('sample');ref=pred[pred.method.eq('group_lasso')&pred.outer_fold.eq(fold)].sort_values('sample')
        for key in ['sample','subject','y_true']:assert np.array_equal(p[key],ref[key]),(name,fold,key)
        assert np.isclose(p.y_true.eq(p.y_pred).mean(),row['accuracy'],atol=1e-12)
        assert im.shape==(4801,4) and np.isfinite(im).all()
        assert np.isclose((im==0).mean(),row['sel_zero_fraction'],atol=1e-12)
        p['method']=name;p['outer_fold']=fold;predictions.append(p)
        maps.setdefault(name,{})[fold]=im
        outer.append(dict(**{k:v for k,v in row.items() if k not in ['method','outer_fold','repro','rule','lambda_g','lambda_sel']},
            method=name,outer_fold=fold,rule=rule,repro=s['repro'],**s['params']))
        sources.append(item)
    out=pd.DataFrame(outer);pr=pd.concat(predictions,ignore_index=True)
    assert len(out)==60 and not pr.duplicated(['sample','method']).any()
    assert pr.groupby(['subject','method']).size().eq(64).all() and pr.subject.nunique()==181
    assert pr.groupby('subject').outer_fold.nunique().eq(1).all()
    acc=pr.assign(correct=pr.y_true.eq(pr.y_pred)).groupby(['subject','method']).correct.mean().rename('accuracy').reset_index().rename(columns={'subject':'unit'})
    acc=report.split_names(acc);out=report.split_names(out)
    stability={}
    for name,parts in maps.items():
        M=np.stack([parts[i] for i in range(5)]).reshape(5,-1)
        stability[name]=float(np.corrcoef(M)[np.triu_indices(5,1)].mean())
    out['stability']=[stability[m if r=='min' else m+'@'+r] for m,r in zip(out.method,out.rule)]
    out['R']=out.repro;out['nonzero']=1-out.sel_zero_fraction
    sums=report.summarize(out,['log_loss','R','stability','nonzero']).drop(columns='n').merge(report.summarize(acc,['accuracy']),on=['rule','method'])
    tests=report.tests(acc,['accuracy'],'wilcox')
    rng=np.random.default_rng(2036);idx=rng.integers(0,181,(10000,181))
    for i,r in tests.iterrows():
        p=acc[acc.rule.eq(r.rule)].pivot(index='unit',columns='method',values='accuracy')
        delta=(p[r.method]-p.group_lasso).to_numpy()
        tests.loc[i,['ci_low','ci_high']]=np.percentile(delta[idx].mean(axis=1),[2.5,97.5])
    for frame,name in [(out,'folds'),(pr,'predictions'),(acc,'accuracy'),(sums,'summary'),(tests,'tests')]:frame.to_csv(OUT/(name+'.csv'),index=False)
    np.savez_compressed(OUT/'maps.npz',**{n:np.stack([v[i] for i in range(5)]) for n,v in maps.items()})
    diagnostic=dict(participants=181,samples=11584,selection_rows=60,new_unique_refits=12,
        rules=['min','1se_repro','1se_fold_repro'],R_source='Selection split-half correlation; not fresh-R',
        states={str(k):int(v) for k,v in out.groupby(['method','common_residual_1e-03']).size().items()},
        original_reports_preserved=True,new_fit_calls=0,precision_refits_used=len(precision_ids),precision_rule_rows=precision_rows)
    if precision:
        assert len(precision_ids)==5 and precision_rows==9
        assert out['common_residual_1e-03'].eq('pass').all()
    (OUT/'sources.json').write_text(json.dumps(sources,indent=2),encoding='utf-8')
    (OUT/'validation.json').write_text(json.dumps(diagnostic,indent=2))
    print(sums[['rule','method','accuracy','log_loss','R','stability','nonzero']].to_string(index=False))
    print(tests[['rule','method','diff','p_holm','ci_low','ci_high']].to_string(index=False))
    # Data-only, independent table; all original main/S9 tables remain untouched.
    render_table(sums,tests,OUT,install=not args.original)

if __name__=='__main__':main()
