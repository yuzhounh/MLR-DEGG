"""Format saved B147 ERP results into supplementary tables; never fit or test."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / 'results/erpcore_n170_zero_grid_b147'
DEST = ROOT / 'manuscript/tables/erp_zero_tables.tex'
TEMP = ROOT / 'tmp/erp_zero_tables'
ORDER = ['group_lasso', 'graphnet', 'graph_tv', 'graph_tv_tm', 'superquadratic']
LABEL = dict(zip(ORDER, ['Group lasso', 'GraphNet', 'Graph TV', 'Temporal graph TV', 'Superquadratic']))
RULES = [('min', r'\emph{min}'), ('1se_repro', r'\emph{1se-repro}')]


def read(input_dir, name):
    with (input_dir / name).open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def signed(value, digits=3, scale=1):
    value = float(value) * scale
    return f'{0:.{digits}f}' if round(value, digits) == 0 else f'{value:+.{digits}f}'


def pvalue(value):
    value = float(value)
    return '$<0.0001$' if value < .0001 else f'${value:.4f}$'


def interval(row, scale, digits):
    effect = signed(row['diff'], digits, scale)
    lo, hi = (float(row[key])*scale for key in ('ci_lower', 'ci_upper'))
    bounds = ',\\,'.join(f'{0 if round(x,digits)==0 else x:.{digits}f}' for x in (lo, hi))
    return f'${effect}\\,[{bounds}]$'


def measure_widths(headers, rows, label):
    # Measure individual rendered lines before allocating columns proportionally.
    choices = [[r'\textbf{' + line + '}' for line in header.split(r'\newline')]
               for header in headers]
    for cells in rows:
        for col, cell in enumerate(cells):
            choices[col].extend(cell.split(r'\newline'))
    commands=[]
    for col, lines in enumerate(choices):
        for line in sorted(set(lines)):
            commands.append(r'\settowidth{\ProbeWidth}{'+line+'}'+
                            rf'\typeout{{ERPWIDTH:{label}:{col}:\the\ProbeWidth}}')
    return commands, len(headers)


def render_tables(input_dir, destination, work_dir):
    validation = json.loads((input_dir/'validation.json').read_text(encoding='utf-8'))
    assert validation['validated'] and not validation['full_expanded_grid_executed']
    summary = {(x['rule'],x['method']):x for x in read(input_dir, 'summary.csv') if x['scenario']=='zero_inclusive_grid'}
    zero = {(x['rule'],x['base']):x for x in read(input_dir, 'zero_selection.csv')}
    comparisons = {(x['rule'],x['method'],x['contrast'],x['metric']):x for x in read(input_dir, 'comparisons.csv')}
    heads4 = ['Method', r'Accuracy (\%)\newlineMean (SD)', 'Log-loss', r'Fresh $R$', 'Stability',
              r'Zero-strength folds\newline$n/185$ (\%)']
    heads5 = ['Method', r'$\Delta$ Accuracy (CI)', r'$p_{\rm Acc}$',
              r'$\Delta R$ (CI)', r'$p_R$', r'$\Delta$ Stability', r'$p_{\rm Stab}$']
    blocks4=[]
    for rule, title in RULES:
        rows=[]
        for method in ORDER:
            s=summary[rule,method];z=zero[rule,method]
            rows.append([LABEL[method], f'{100*float(s["accuracy"]):.2f} ({100*float(s["accuracy_sd"]):.2f})',
                         f'{float(s["log_loss"]):.3f}', f'{float(s["R"]):.3f}', f'{float(s["stability"]):.3f}',
                         'Fixed at zero' if method=='group_lasso' else f'{int(z["zero"])}/185 ({100*float(z["zero_fraction"]):.1f}\\%)'])
        blocks4.append((title+' rule',rows))
    blocks5=[]
    for contrast, name in [('versus_group_lasso','Group lasso'),('versus_original_same_method','the same member\'s original search')]:
        for rule,title in RULES:
            rows=[]
            for method in ORDER[1:]:
                a,r,s=(comparisons[rule,method,contrast,metric] for metric in ('accuracy','R','stability'))
                rows.append([LABEL[method], interval(a,100,2), pvalue(a['p_holm']), interval(r,1,3),
                             pvalue(r['p_holm']), '$'+signed(s['diff'])+'$', pvalue(s['p_holm'])])
            blocks5.append((title+' versus '+name,rows))

    commands4,_=measure_widths(heads4,[r for _,rows in blocks4 for r in rows],'erp_performance')
    commands5,_=measure_widths(heads5,[r for _,rows in blocks5 for r in rows],'erp_contrasts')
    work_dir.mkdir(parents=True,exist_ok=True)
    probe=work_dir/'erp_zero_widths.tex'
    probe.write_text('\n'.join([r'\documentclass[11pt]{article}',r'\usepackage[T1]{fontenc}',r'\usepackage{amsmath}',
        r'\renewcommand{\rmdefault}{lmr}',r'\newlength{\ProbeWidth}',r'\begin{document}',
        r'\fontsize{9pt}{11pt}\selectfont',*commands4,*commands5,r'\end{document}']),encoding='utf-8')
    compiler=shutil.which('pdflatex')
    if not compiler:
        raise RuntimeError('pdflatex is required for the table-width probe')
    result=subprocess.run([compiler,'-interaction=nonstopmode','-halt-on-error',probe.name],cwd=work_dir,capture_output=True,text=True,encoding='utf-8',errors='replace')
    (work_dir/'erp_zero_width_probe.txt').write_text(result.stdout+result.stderr,encoding='utf-8')
    if result.returncode:
        raise RuntimeError('Table-width measurement failed; see temporary probe log')
    measurements={name:[0.]*count for name,count in [('erp_performance',6),('erp_contrasts',7)]}
    for name,col,width in re.findall(r'ERPWIDTH:(erp_performance|erp_contrasts):(\d+):([\d.]+)pt',result.stdout):
        measurements[name][int(col)]=max(measurements[name][int(col)],float(width))
    fractions={name:[v/sum(widths) for v in widths] for name,widths in measurements.items()}

    def columns(name):
        length=r'\ErpZeroContentWidth' if name=='erp_performance' else r'\ErpZeroComparisonContentWidth'
        return '\n'.join((r'>{\raggedright\arraybackslash}' if j==0 else r'>{\centering\arraybackslash}')+
                         'p{'+f'{x:.9f}'+length+'}' for j,x in enumerate(fractions[name]))

    def header(cells):
        return ' & '.join(r'\ManuscriptHeader{'+x.replace(r'\newline',r'\newline{}')+'}' for x in cells)+r' \\'

    def block_rows(blocks,count):
        lines=[]
        for index,(title,rows) in enumerate(blocks):
            if index: lines.append(r'\midrule')
            lines.append(r'\multicolumn{'+str(count)+r'}{l}{'+title+(r'} \\*' if count==7 else r'} \\'))
            lines.extend(' & '.join(cells)+(r' \\*' if count==7 and i<len(rows)-1 else r' \\')
                         for i,cells in enumerate(rows))
        return lines

    caption4=(r'\textbf{ERP performance when zero graph strength is allowed.} '
        r'37 participants; accuracy is the participant mean (standard deviation, SD), in \%. '
        r'Log-loss and fresh split-half $R$ are participant means over five outer folds. '
        r'Stability is the mean cross-fold importance-map correlation. '
        r'The last column counts outer-fold searches selecting zero graph strength; Group lasso fixes it at zero. '
        r'The four graph-selection procedures can return the group-lasso objective.')
    caption5=(r'\textbf{Paired comparisons for the ERP zero-strength sensitivity.} '
        r'Differences are the zero-inclusive procedure minus the reference named in each group. '
        r'Accuracy differences and their confidence intervals (CIs) are in percentage points; '
        r'Fresh $R$ and stability differences use their original scales. '
        r'Pointwise 95\% CIs for mean differences use 10,000 paired participant bootstrap resamples. '
        r'$p$-values are paired Wilcoxon values, Holm-corrected over the four graph members within each rule, contrast and metric; '
        r'this differs from the six-comparison families in the main analysis. '
        r'The mean-difference intervals and adjusted rank-test $p$-values need not give identical decisions. '
        r'$n=37$ for every comparison; exact zero differences are omitted from the signed-rank tests.')
    lines=[r'% Generated only from saved B147 CSVs; no fitting or statistical calculations.',
        r'\newlength{\ErpZeroContentWidth}',r'\setlength{\ErpZeroContentWidth}{\dimexpr\textwidth-12\tabcolsep\relax}',
        r'\begin{table}[!htbp]',r'\centering',r'\caption{'+caption4+r'}\label{tab:erp-zero}',
        r'\ManuscriptTableStyle',r'\begin{tabular}{',columns('erp_performance'),'}',r'\toprule',header(heads4),r'\midrule',
        *block_rows(blocks4,6),r'\bottomrule',r'\end{tabular}',r'\end{table}',r'\FloatBarrier',
        r'\newlength{\ErpZeroComparisonContentWidth}',r'\setlength{\ErpZeroComparisonContentWidth}{\dimexpr\textwidth-14\tabcolsep\relax}',
        r'\setlength{\LTcapwidth}{\textwidth}',r'\setlength{\LTleft}{\fill}',r'\setlength{\LTright}{\fill}',
        r'\begin{longtable}{',columns('erp_contrasts'),'}',r'\caption{'+caption5+r'}\label{tab:erp-zero-contrasts} \\',
        r'\toprule',header(heads5),r'\midrule',r'\endfirsthead',
        r'\multicolumn{7}{l}{\textbf{Table \thetable.} \emph{Continued}} \\',
        r'\toprule',header(heads5),r'\midrule',r'\endhead',r'\endfoot',r'\bottomrule',r'\endlastfoot',
        *block_rows(blocks5,7),r'\end{longtable}']
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    manifest=dict(inputs={name:hashlib.sha256((input_dir/name).read_bytes()).hexdigest() for name in
                         ('summary.csv','comparisons.csv','zero_selection.csv','validation.json')},
        output=str(destination),rows={'erp_performance':10,'erp_contrasts':16},measured_widths_pt=measurements,column_fractions=fractions,
        no_fitting=True,no_statistical_recalculation=True)
    (work_dir/'table_formatting.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(dict(output=str(destination),rows=manifest['rows'],fractions=fractions),indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.input_dir or args.output_dir:
        if not (args.input_dir and args.output_dir):
            parser.error("Specify --input-dir and --output-dir together")
        render_tables(args.input_dir, args.output_dir / DEST.name, args.output_dir / "table widths")
    else:
        render_tables(INPUT, DEST, TEMP)


if __name__=='__main__':
    main()
