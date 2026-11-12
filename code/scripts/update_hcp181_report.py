"""Install HCP181 reporting products from verified results or compact archived inputs. No fitting."""
import ast
import importlib
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import _bootstrap
from mlrdegg.paths import PROJECT_ROOT as ROOT
from mlrdegg.experiment.provenance import fingerprint, sha256
report=importlib.import_module('paper_report')

def ptex(p):
    return '$<0.0001$' if p<.0001 else f'{p:.4f}'

def render_primary(dest,inputs):
    pa=pd.read_csv(inputs/'hcp_primary_accuracy.csv');pf=pd.read_csv(inputs/'hcp_primary_folds.csv');pt=pd.read_csv(inputs/'hcp_primary_tests.csv')
    ps=report.summarize(pa,['accuracy']).merge(report.summarize(pf,['log_loss','stability']),on=['rule','method'],suffixes=('','_fold'))
    ps['accuracy_pct']=100*ps.accuracy;ps['accuracy_sd_pct']=100*ps.accuracy_sd
    ps[['rule','method','accuracy_pct','accuracy_sd_pct','log_loss','stability']].to_csv(dest/'common_grid_summary.csv',index=False)
    pt.assign(diff_pp=100*pt['diff'],ci_low_pp=100*pt.ci_low,ci_high_pp=100*pt.ci_high).to_csv(dest/'common_grid_tests.csv',index=False)
    selected=[];mainrows=[];supprows=[]
    for rule in report.RULES:
        mainrows.append(r'\multicolumn{6}{l}{\emph{'+rule.replace('_','-')+r'} rule} \\')
        for method in ['group_lasso','graphnet','graph_tv','superquadratic']:
            r=ps[(ps.rule==rule)&(ps.method==method)].iloc[0]
            delta,p='--','--'
            if method!='group_lasso':
                t=pt[(pt.rule==rule)&(pt.method==method)].iloc[0]
                delta=f'{100*t["diff"]:.2f} ({100*t.ci_low:.2f}--{100*t.ci_high:.2f})';p=ptex(t.p_holm)
            cells=[report.LABEL[method],f'{r.accuracy_pct:.1f} ({r.accuracy_sd_pct:.1f})',delta,p,f'{r.log_loss:.3f}',f'{r.stability:.2f}']
            mainrows.append(' & '.join(cells)+r' \\')
            chosen=pf[(pf.rule==rule)&(pf.method==method)]
            params=[ast.literal_eval(s) for s in chosen.selected]
            grid=lambda key:'/'.join(f'{v:g}' for v in sorted({s[key] for s in params if key in s})) or '--'
            lead=r'\emph{'+rule.replace('_','-')+'}' if method=='group_lasso' else ''
            supprows.append(' & '.join([lead]+cells+[grid('lambda_sel'),grid('lambda_g')])+r' \\')
            for row,pars in zip(chosen.to_dict('records'),params):
                selected.append({'rule':rule,'method':method,'outer_fold':row['outer_fold'],**pars,
                    'numerical_stopping':row['numerical_stopping'],'iteration_limit_hit':row['iteration_limit_hit'],
                    **{key:row.get(key) for key in ['common_residual','common_residual_error_bound',
                        'common_residual_1e-03','reported_solution_source']}})
        if rule=='min':mainrows.append(r'\midrule')
    (dest/'common_grid_rows.tex').write_text('\\newcommand{\\HcpCommonRows}{%\n'+'\n'.join(mainrows)+'\n}\n')
    (dest/'common_grid_detail_rows.tex').write_text('\\newcommand{\\HcpCommonDetailRows}{%\n'+'\n'.join(supprows)+'\n}\n')
    pd.DataFrame(selected).to_csv(dest/'common_grid_selection.csv',index=False)


def main(from_archive=False):
    dest=ROOT/'manuscript/tables';inputs=dest/'inputs';run=ROOT/'results/hcp181_merged';rpt=run/'report'
    manifest_path=dest/'report_manifest.json';manifest=json.loads(manifest_path.read_text())
    if from_archive:
        assert manifest['input_sha256']==fingerprint(sorted(inputs.glob('*.csv')))
    else:
        validation=json.loads((rpt/'validation.json').read_text())
        assert validation['participants']==181 and validation['external_tasks']==85
        copies={'family_accuracy.csv':'hcp_accuracy.csv','family_folds.csv':'hcp_folds.csv',
                'primary_accuracy.csv':'hcp_primary_accuracy.csv','primary_folds.csv':'hcp_primary_folds.csv',
                'primary_tests.csv':'hcp_primary_tests.csv','external_baselines.csv':'hcp_external_baselines.csv'}
        for source,target in copies.items():shutil.copy2(rpt/source,inputs/target)
        shutil.copy2(rpt/'validation.json',dest/'hcp181_validation.json')
        shutil.copy2(rpt/'unfinished_rows.csv',dest/'numerical/hcp181_unfinished_rows.csv')
        for rel in list(manifest['original_result_sha256']):
            if rel.startswith('results/hcp_wm_surf'):del manifest['original_result_sha256'][rel]
        sources=[]
        for directory in ['family','primary']:
            d=run/directory
            sources += [d/'config.yaml',d/'outer_results.csv',d/'inner_scores.csv',d/'units/pooled_maps.npz',d/'units/pooled_predictions.csv']
        sources += [run/'family/fresh_repro_seed5003.csv',run/'external/outer_results.csv',run/'external/test_predictions.csv']
        manifest['original_result_sha256'].update(fingerprint(sources))
    acc=pd.read_csv(inputs/'hcp_accuracy.csv');folds=pd.read_csv(inputs/'hcp_folds.csv')
    table=report.summarize(acc,['accuracy']).merge(report.summarize(folds,['log_loss','R','stability','zeros']),on=['rule','method'],suffixes=('','_fold'))
    paired=report.tests(acc,['accuracy'],'wilcox')
    table.to_csv(dest/'hcp.csv',index=False);paired.to_csv(dest/'hcp_tests.csv',index=False)
    (dest/'hcp_rows.tex').write_text('\\newcommand{\\HcpTableRows}{%\n'+report.tex_rows('hcp',table,paired)+'}\n')
    render_primary(dest,inputs)
    ext=pd.read_csv(inputs/'hcp_external_baselines.csv')
    oldext=pd.read_csv(dest/'external_baselines.csv')
    pd.concat([oldext[oldext.dataset.ne('HCP')],ext],ignore_index=True).to_csv(dest/'external_baselines.csv',index=False)
    extrows=[]
    for name,label in [('lda_shrinkage','Shrinkage LDA'),('svm_linear','Linear SVM')]:
        g=ext[ext.baseline.eq(name)].set_index('reference');r=g.loc['group_lasso'];s=g.loc['graphnet']
        extrows.append(' & '.join(['HCP' if name=='lda_shrinkage' else '',label,f'{r.baseline_accuracy_pct:.1f} ({r.baseline_accuracy_sd_pct:.1f})',f'${r.diff_pp:.2f}$',f'{int(r.wins)}/{int(r.losses)}',ptex(r.p_holm),'',f'${s.diff_pp:.2f}$',f'{int(s.wins)}/{int(s.losses)}',ptex(s.p_holm)])+r' \\')
    (dest/'hcp_external_rows.tex').write_text('\\newcommand{\\HcpExternalRows}{%\n'+'\n'.join(extrows)+'\n}\n')
    # Update only HCP in the already-present zero-convention sensitivity, preserving ERP/simulation.
    alt=report.tests(acc,['accuracy'],'zsplit')
    zero=paired.merge(alt[['rule','method','metric','p_holm']],on=['rule','method','metric'],suffixes=('_reported','_alternative'))
    zero.insert(0,'dataset','hcp');zero['zero_differences']=zero.n-zero.wins-zero.losses
    zero['reported_convention']='wilcox';zero['alternative_convention']='zsplit'
    zero['same_decision_at_0.05']=(zero.p_holm_reported<.05)==(zero.p_holm_alternative<.05)
    old=pd.read_csv(dest/'wilcoxon_zero_sensitivity.csv')
    pd.concat([old[old.dataset.ne('hcp')],zero],ignore_index=True).to_csv(dest/'wilcoxon_zero_sensitivity.csv',index=False)
    manifest['active_hcp_run']='hcp181_merged'
    manifest['units']['hcp_accuracy']='181 unrelated participants; 64 segments each'
    manifest['fresh_R_quality']='HCP181: all recorded fresh half fits reached numerical stopping; ERP37 reused 27-participant fresh-half states remain unknown.'
    manifest['input_sha256']=fingerprint(sorted(inputs.glob('*.csv')))
    manifest['report_code_sha256']=sha256(Path(__file__))
    details=manifest.get('hcp181_reporting',{})
    details.update({'primary':'common-grid four-member comparison with accepted outer precision' if 'primary_outer_refinement' in details else 'common-grid four-member comparison',
                    'family':'method-specific seven-method comparison','no_refitting':True,
                    'unfinished_outer_refits':1,'unfinished_outer_refits_scope':'historical full-family comparison'})
    manifest['hcp181_reporting']=details
    manifest_path.write_text(json.dumps(manifest,indent=2))
    print(json.dumps({'participants':acc.unit.nunique(),'hcp_table_rows':len(table),'zero_convention_changes':int((~zero['same_decision_at_0.05']).sum()),'hcp_zero_differences':int(zero.zero_differences.sum()),'from_archive':from_archive}))

if __name__=='__main__':main()
