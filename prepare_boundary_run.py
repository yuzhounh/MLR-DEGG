"""Prepare paper-specific sensitivity plans from local results; never launch fits."""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--phase', choices=('inner', 'half', 'refit'), required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    spec = importlib.util.spec_from_file_location('boundary_selection', root/'code/scripts/33_boundary_selection.py')
    selection = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(selection)
    selection.ROOT = root
    sys.path.insert(0, str(root/'code/src'))
    run = root/'results/hcp181_boundary_b122'
    phase = run/args.phase
    if phase.exists():
        raise RuntimeError(f'Existing stage retained: {phase}')

    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2, default=lambda x: x.item() if hasattr(x, 'item') else str(x)), encoding='utf-8')

    def summaries(kind):
        plan = json.loads((run/'plans'/f'{kind}_partition_a.json').read_text(encoding='utf-8'))
        rows = []
        for task in plan:
            task_dir = run/'tasks'/task['id']
            receipt = json.loads((task_dir/'completion.json').read_text(encoding='utf-8'))
            if receipt['state'] != 'completed':
                raise RuntimeError(f'Incomplete task: {task["id"]}')
            rows.append(json.loads((task_dir/'summary.json').read_text(encoding='utf-8')))
        return rows

    if args.phase == 'inner':
        config, tasks = selection.make_plan()
        inputs = run/'inputs'
        if inputs.exists() or (run/'runtime').exists():
            raise RuntimeError('Existing inputs/runtime retained')
        inputs.mkdir(parents=True)
        (inputs/'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
        runtime = run/'runtime/code'
        shutil.copytree(root/'code/src', runtime/'src')
        (runtime/'scripts').mkdir()
        for name in ('_bootstrap.py', '03_run_decoding.py'):
            shutil.copy2(root/'code/scripts'/name, runtime/'scripts'/name)
    else:
        config = yaml.safe_load((run/'inputs/config.yaml').read_text(encoding='utf-8'))
        rows = selection.old_rows(config)
        for row in summaries('inner'):
            task = row['task']
            key = selection.identity(task['method'], task['outer_fold'], task['params'])
            rows.setdefault(key, []).append(dict(**row['metrics'], inner_fold=task['inner_fold']))
        states = selection.statistics(config, rows)
        if args.phase == 'half':
            tasks = selection.needed_halves(states)
        else:
            chosen = selection.select(states, summaries('half'))
            tasks, manifest = selection.plan_refits(config, chosen)
            write(run/'selection.json', chosen)
            write(run/'refit_manifest.json', manifest)
    plan = run/'plans'/f'{args.phase}_partition_a.json'
    if plan.exists():
        raise RuntimeError(f'Existing plan retained: {plan}')
    write(plan, tasks)
    print(json.dumps({'phase': args.phase, 'tasks': len(tasks), 'plan': str(plan), 'fits_started': 0}))


if __name__ == '__main__':
    main()
