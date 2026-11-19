"""Selection and reporting logic from the retained ERP zero-grid experiment."""
import ast
import hashlib
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import wilcoxon
from mlrdegg.experiment.cv import grid_statistics, select_grid_point
from mlrdegg.experiment.models import grid_points

METHODS = ['group_lasso', 'graphnet', 'graph_tv', 'graph_tv_tm', 'superquadratic']
RULES = ['min', '1se_repro']
METRICS = ['accuracy', 'log_loss', 'R', 'stability', 'nonzero_fraction', 'selection_R']


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.new')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    tmp.replace(path)

def stamp():
    return datetime.now(timezone.utc).isoformat()

def signature(unit, fold, method, params):
    # The no-graph convex objective is represented once, by group lasso.
    g = float(params.get('lambda_g', 0))
    base = 'group_lasso' if g == 0 else method
    normalized = {'lambda_sel': float(params['lambda_sel'])}
    if g:
        normalized['lambda_g'] = g
    payload = [str(unit), int(fold), base, normalized]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]

def read_sources(source):
    cfg = yaml.safe_load((source / 'config.yaml').read_text(encoding='utf-8'))
    outer = pd.read_csv(source / 'outer_results.csv', low_memory=False)
    inner = pd.read_csv(source / 'inner_scores.csv', low_memory=False)
    fresh = pd.read_csv(source / 'fresh_repro_seed5003.csv')
    outer['base'] = outer.method.str.split('@').str[0]
    outer = outer[outer.base.isin(METHODS)].copy()
    inner = inner[inner.method.isin(METHODS)].copy()
    assert outer.unit.nunique() == 37 and len(outer) == 1850 and len(inner) == 46620
    assert not outer.error.fillna('').ne('').any()
    assert not inner.error.fillna('').ne('').any()
    assert not outer.duplicated(['unit', 'outer_fold', 'method']).any()
    return cfg, outer, inner, fresh

def select_with_zero(cfg, outer, inner):
    definitions = {m['name']: m for m in cfg['experiment']['methods'] if m['name'] in METHODS}
    indexed_outer = outer.set_index(['unit', 'outer_fold', 'base', 'rule'])
    selections, checked = [], 0
    for unit, unit_rows in inner.groupby('unit', sort=True):
        records = unit_rows.fillna({'error': ''}).to_dict('records')
        for fold in range(5):
            all_stats = {}
            for method in METHODS:
                definition = definitions[method]
                stats = grid_statistics(records, fold, definition)
                repro = unit_rows[(unit_rows.outer_fold == fold) & (unit_rows.method == method)]
                for gi, state in enumerate(stats):
                    source = repro[repro.grid_index == gi]
                    assert len(source) == 3 and source.inner_fold.nunique() == 3
                    assert source.repro.notna().all() and source.repro.nunique() == 1
                    state['repro'] = float(source.repro.iloc[0])
                all_stats[method] = stats
                for rule in RULES:
                    old = grid_points(definition)[select_grid_point(stats, rule)]
                    actual = ast.literal_eval(indexed_outer.loc[(unit, fold, method, rule), 'selected'])
                    assert old == actual, (unit, fold, method, rule, old, actual)
                    checked += 1
            for method in METHODS:
                zero_params = grid_points(definitions['group_lasso'])
                candidates = [('group_lasso', p, s) for p, s in zip(zero_params, all_stats['group_lasso'])]
                if method != 'group_lasso':
                    candidates += [(method, p, s) for p, s in zip(grid_points(definitions[method]), all_stats[method])]
                for rule in RULES:
                    chosen = select_grid_point([x[2] for x in candidates], rule)
                    base, params, stats = candidates[chosen]
                    name = method if rule == 'min' else method + '@' + rule
                    selections.append(dict(unit=unit, outer_fold=fold, method=name, base=method,
                        rule=rule, canonical_method=base, params=params,
                        target_id=signature(unit, fold, base, params),
                        lambda_sel=float(params['lambda_sel']), lambda_g=float(params.get('lambda_g', 0)),
                        selection_R=float(stats['repro']), inner_accuracy=float(stats['accuracy']),
                        inner_mean_loss=float(-stats['score']), inner_pooled_loss=float(stats['log_loss']),
                        inner_se=float(stats['se'])))
    assert checked == 1850
    return selections, checked

def participant_metrics(directory, outer, fresh):
    outer=outer.copy()
    outer['base']=outer.method.str.split('@').str[0]
    outer=outer[outer.base.isin(METHODS)]
    frame=outer.drop(columns=['fresh_repro'],errors='ignore').merge(
        fresh[['unit','outer_fold','method','fresh_repro']],on=['unit','outer_fold','method'],validate='one_to_one')
    frame['R']=frame.fresh_repro
    frame['selection_R']=frame.repro
    stable=[];zeros=[]
    for unit in sorted(frame.unit.unique()):
        with np.load(directory/'units'/(unit+'_maps.npz')) as z:
            for name in frame.loc[frame.unit==unit,'method'].unique():
                maps=z[name]
                flat=maps.reshape(5,-1)
                correlations=[np.corrcoef(flat[a],flat[b])[0,1] for a,b in itertools.combinations(range(5),2)
                              if flat[a].std()>0 and flat[b].std()>0]
                stable.append(dict(unit=unit,method=name,stability=float(np.mean(correlations)) if correlations else np.nan))
                zeros.extend(dict(unit=unit,method=name,outer_fold=f,nonzero_fraction=float(np.mean(maps[f]!=0))) for f in range(5))
    frame=frame.merge(pd.DataFrame(stable),on=['unit','method'],validate='many_to_one')
    frame=frame.merge(pd.DataFrame(zeros),on=['unit','method','outer_fold'],validate='one_to_one')
    result=frame.groupby(['unit','method'])[METRICS].mean().reset_index()
    result['base']=result.method.str.split('@').str[0]
    result['rule']=result.method.map(lambda s:s.partition('@')[2] or 'min')
    assert len(result)==370 and result.unit.nunique()==37
    return result

def holm(values):
    p=np.asarray(values);order=np.argsort(p);adj=np.empty_like(p);running=0.
    for rank,i in enumerate(order):
        running=max(running,(len(p)-rank)*p[i]);adj[i]=min(running,1.)
    return adj

def analyze(source, retained, output, old_outer, old_fresh, manuscript_reference=None):
    completion=json.loads((retained/'completion.json').read_text(encoding='utf-8'))
    assert completion['state']=='completed'
    old=participant_metrics(source,old_outer,old_fresh)
    new=participant_metrics(retained,pd.read_csv(retained/'outer_results.csv',low_memory=False),pd.read_csv(retained/'fresh_repro_seed5003.csv'))
    old['scenario']='original_positive_grid';new['scenario']='zero_inclusive_grid'
    all_units=pd.concat([old,new],ignore_index=True)
    all_units.to_csv(output/'participant_metrics.csv',index=False)
    summaries=[]
    for (scenario,rule,method),q in all_units.groupby(['scenario','rule','base']):
        row=dict(scenario=scenario,rule=rule,method=method,n=len(q))
        for metric in METRICS:
            row[metric]=float(q[metric].mean());row[metric+'_sd']=float(q[metric].std(ddof=1))
        summaries.append(row)
    pd.DataFrame(summaries).to_csv(output/'summary.csv',index=False)
    # Baseline and original manuscript table replay checks use unrounded saved values.
    if manuscript_reference is not None:
        reference=pd.read_csv(manuscript_reference)
        for row in reference[reference.method.isin(METHODS)].itertuples():
            q=old[(old.base==row.method)&(old.rule==row.rule)]
            for metric in ('accuracy','log_loss','R','stability'):
                assert np.isclose(q[metric].mean(),getattr(row,metric),atol=1e-12,rtol=0),(row.method,row.rule,metric)
    og=old[old.base=='group_lasso'].set_index(['unit','rule'])
    ng=new[new.base=='group_lasso'].set_index(['unit','rule'])
    assert np.allclose(og[METRICS].sort_index(),ng[METRICS].sort_index(),atol=1e-12,equal_nan=True)
    comparisons=[]
    rng=np.random.default_rng(20261115)
    for rule in RULES:
        baseline=new[(new.base=='group_lasso')&(new.rule==rule)].set_index('unit')
        for method in METHODS[1:]:
            current=new[(new.base==method)&(new.rule==rule)].set_index('unit')
            previous=old[(old.base==method)&(old.rule==rule)].set_index('unit')
            for contrast,reference_frame in [('versus_group_lasso',baseline),('versus_original_same_method',previous)]:
                for metric in ('accuracy','R','log_loss','stability'):
                    d=(current[metric]-reference_frame[metric]).dropna().to_numpy()
                    draws=d[rng.integers(0,len(d),size=(10000,len(d)))].mean(axis=1)
                    lo,hi=np.quantile(draws,[.025,.975])
                    p=float(wilcoxon(d,zero_method='wilcox').pvalue) if np.any(d!=0) else 1.
                    comparisons.append(dict(rule=rule,method=method,contrast=contrast,metric=metric,n=len(d),
                        diff=float(d.mean()),ci_lower=float(lo),ci_upper=float(hi),p=p,
                        wins=int((d>0).sum()),losses=int((d<0).sum()),ties=int((d==0).sum())))
    comparisons=pd.DataFrame(comparisons)
    for _,q in comparisons.groupby(['rule','contrast','metric']):
        comparisons.loc[q.index,'p_holm']=holm(q.p.to_numpy())
        comparisons.loc[q.index,'family_size']=len(q)
    comparisons.to_csv(output/'comparisons.csv',index=False)
    selections=pd.read_csv(retained/'selection.csv')
    zero=selections.groupby(['base','rule']).lambda_g.agg(folds='size',zero=lambda s:int((s==0).sum())).reset_index()
    zero['zero_fraction']=zero.zero/zero.folds
    zero.to_csv(output/'zero_selection.csv',index=False)
    write_json(output/'analysis_receipt.json',dict(created_utc=stamp(),subjects=37,outer_folds_per_subject=5,
        bootstrap_replicates=10000,bootstrap_seed=20261115,
        pairing='Participant means over five outer folds; folds are not independent test units.',
        multiplicity='Holm over four graph members within rule, contrast and metric. Sensitivity analysis; not correction across all endpoints/rules.',
        intervals='Pointwise participant paired percentile bootstrap 95% intervals.',
        original_summary_replayed=True,group_lasso_unchanged=True,no_manuscript_edits=True,
        new_fit_quality=pd.read_csv(retained/'new_fit_log.csv').groupby('stage').converged.agg(['size','sum']).to_dict('index')))
    print(pd.DataFrame(summaries)[['scenario','rule','method','accuracy','R','stability']].to_string(index=False))
    print(comparisons[comparisons.metric.isin(['accuracy','R'])][['rule','method','contrast','metric','diff','ci_lower','ci_upper','p_holm']].to_string(index=False))
