"""Analyze retained B122 outputs only; preserve the original HCP comparisons."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import argparse,importlib,json
from io import BytesIO
from zipfile import ZipFile
from pathlib import Path
import numpy as np,pandas as pd
import _bootstrap
from paper_result_policy import original_results_active, primary_source, boundary_source
from hcp_common_grid import MEMBERS, check_original_metrics, paired_accuracy_tests, require_original_primary
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'results/hcp181_boundary_b122'
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
    if not original_results_active():
        raise RuntimeError('Explicit original results are required for the expanded-grid table')
    saved=boundary_source()
    render_table(pd.read_csv(saved/'summary.csv'),pd.read_csv(saved/'tests.csv'),None)

def load_original_refits(entries, inputs):
    expected = {}
    for item in entries:
        source, selection = item["source"], item["selection"]
        if source["kind"] == "new_refit":
            task_id = source["task_id"]
            key = (selection["method"], selection["outer_fold"], selection["params"])
            if task_id in expected and expected[task_id] != key:
                raise ValueError(f"Inconsistent selections for {task_id}")
            expected[task_id] = key
        elif source["kind"] != "active_primary":
            raise ValueError(f"Unknown refit source: {source['kind']}")
    if any(task_id in {".", ".."} or Path(task_id).name != task_id or
           any(c in task_id for c in "\\\\/:") for task_id in expected):
        raise ValueError("Invalid refit task identifier")

    files = ["summary.json", "predictions.csv", "importance.npz"]
    found = {}

    def add(task_id, parts, source):
        if task_id in found:
            raise ValueError(f"Duplicate retained output for {task_id}")
        summary = json.loads(parts[0])
        task, metrics = summary["task"], summary["metrics"]
        check_original_metrics(metrics, source)
        if "source" in summary or any(key.startswith("precision_") for key in metrics):
            raise ValueError(f"Unexpected refinement output for {task_id} in {source}")
        actual = task["method"], task["outer_fold"], task["params"]
        if task["id"] != task_id or task["kind"] != "refit" or actual != expected[task_id]:
            raise ValueError(f"Task does not match the refit manifest: {task_id}")
        predictions = pd.read_csv(BytesIO(parts[1]))
        with np.load(BytesIO(parts[2])) as arrays:
            found[task_id] = metrics, predictions, arrays["importance"]

    for path in inputs:
        if path.is_dir():
            for task_id in expected:
                task_dir = path / task_id
                if task_dir.is_dir():
                    add(task_id, [(task_dir / f).read_bytes() for f in files], path)
        elif path.is_file():
            with ZipFile(path) as archive:
                names = archive.namelist()
                for task_id in expected:
                    targets = [f"{task_id}/{name}" for name in files]
                    if targets[0] in names:
                        if any(names.count(name) != 1 for name in targets):
                            raise ValueError(f"Missing or repeated files for {task_id} in {path}")
                        add(task_id, [archive.read(name) for name in targets], path)
        else:
            raise FileNotFoundError(path)
    missing = expected.keys() - found.keys()
    if missing:
        raise ValueError(f"Missing original outer-refit outputs: {', '.join(sorted(missing))}")
    return found


def analyze_boundary(base, manifest, out_dir, *, refits, install=False):
    old=pd.read_csv(base/'outer_results.csv');pred=pd.read_csv(base/'units/pooled_predictions.csv')
    with np.load(base/'units/pooled_maps.npz') as z:oldmaps={k:z[k] for k in z.files}
    outer=[];predictions=[];maps={};sources=[]
    for item in manifest:
        s=item['selection'];src=item['source'];m=s['method'];fold=s['outer_fold'];rule=s['rule']
        name=m if rule=='min' else m+'@'+rule
        if src['kind']=='active_primary':
            selected = old[old.method.eq(src["method"]) & old.outer_fold.eq(fold)]
            assert len(selected) == 1 and src["method"].split("@")[0] == m and src["outer_fold"] == fold
            row = selected.iloc[0].to_dict()
            assert all(np.isclose(row[key], value) for key, value in s["params"].items())
            p=pred[pred.method.eq(src['method'])&pred.outer_fold.eq(fold)].copy()
            im=oldmaps[src['method']][fold]
        else:
            tid=src['task_id']
            row, p, im = refits[tid]
            p = p.copy()
        p=p.sort_values('sample');ref=pred[pred.method.eq('group_lasso')&pred.outer_fold.eq(fold)].sort_values('sample')
        for key in ['sample','subject','y_true']:assert np.array_equal(p[key],ref[key]),(name,fold,key)
        if 'outer_fold' in p:
            assert p.outer_fold.eq(fold).all(), (name, fold)
        assert np.isclose(p.y_true.eq(p.y_pred).mean(),row['accuracy'],atol=1e-12)
        assert im.shape==(4801,4) and np.isfinite(im).all()
        assert np.isclose((im==0).mean(),row['sel_zero_fraction'],atol=1e-12)
        p['method']=name;p['outer_fold']=fold;predictions.append(p)
        maps.setdefault(name,{})[fold]=im
        outer.append(dict(**{k:v for k,v in row.items() if k not in ['method','outer_fold','repro','rule','lambda_g','lambda_sel']},
            method=name,outer_fold=fold,rule=rule,repro=s['repro'],**s['params']))
        sources.append({**item, "source": {
            key: value for key, value in src.items() if key != "precision_source"
        }})
    out=pd.DataFrame(outer);pr=pd.concat(predictions,ignore_index=True)
    expected_names = set(MEMBERS) | {f"{m}@{rule}" for m in MEMBERS for rule in ("1se_repro", "1se_fold_repro")}
    assert set(out.method) == expected_names and not out.duplicated(["method", "outer_fold"]).any()
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
    tests = paired_accuracy_tests(acc)
    out_dir.mkdir(parents=True, exist_ok=True)
    for frame,name in [(out,'folds'),(pr,'predictions'),(acc,'accuracy'),(sums,'summary'),(tests,'tests')]:frame.to_csv(out_dir/(name+'.csv'),index=False)
    np.savez_compressed(out_dir/'maps.npz',**{n:np.stack([v[i] for i in range(5)]) for n,v in maps.items()})
    diagnostic=dict(participants=181,samples=11584,selection_rows=60,new_unique_refits=len(refits),
        rules=['min','1se_repro','1se_fold_repro'],R_source='Selection split-half correlation; not fresh-R',
        states={str(k):int(v) for k,v in out.groupby(['method','common_residual_1e-03']).size().items()},
        original_reports_preserved=True,new_fit_calls=0,precision_refits_used=0,precision_rule_rows=0)
    (out_dir/'sources.json').write_text(json.dumps(sources,indent=2),encoding='utf-8')
    (out_dir/'validation.json').write_text(json.dumps(diagnostic,indent=2))
    print(sums[['rule','method','accuracy','log_loss','R','stability','nonzero']].to_string(index=False))
    print(tests[['rule','method','diff','p_holm','ci_low','ci_high']].to_string(index=False))
    # Data-only, independent table; all original main/S9 tables remain untouched.
    render_table(sums,tests,out_dir,install=install)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-dir", type=Path, help="Original common-grid replay directory")
    parser.add_argument("--refit-manifest", type=Path, help="Retained expanded-grid refit_manifest.json")
    parser.add_argument("--refit-input", action="append", type=Path, help="Original refit outputs ZIP or directory (repeatable)")
    parser.add_argument("--output-dir", type=Path, help="Empty destination for replayed reports and table rows")
    parser.add_argument("--original", action="store_true", help="Legacy local replay without installing a manuscript table")
    args = parser.parse_args()
    portable = (args.primary_dir, args.refit_manifest, args.refit_input, args.output_dir)
    if any(portable):
        if not all(portable):
            parser.error("Specify --primary-dir, --refit-manifest, --refit-input and --output-dir together")
        if args.output_dir.exists() and any(args.output_dir.iterdir()):
            parser.error("The output directory must be empty")
        require_original_primary(args.primary_dir)
        manifest = json.loads(args.refit_manifest.read_text(encoding="utf-8"))
        analyze_boundary(args.primary_dir, manifest, args.output_dir,
                         refits=load_original_refits(manifest, args.refit_input))
        return
    if not original_results_active():
        parser.error("Provide the original retained outputs with --primary-dir, --refit-manifest, --refit-input and --output-dir")
    manifest = json.loads((RUN/"refit_manifest.json").read_text(encoding="utf-8"))
    collected = RUN / "collected"
    refits = load_original_refits(manifest, sorted(path for path in collected.glob("refit_*") if path.is_dir()))
    analyze_boundary(primary_source(), manifest, boundary_source(), refits=refits, install=not args.original)


if __name__ == "__main__":
    main()
