"""Select retained experimental outputs for manuscript reporting, without fitting."""
import json
from mlrdegg.paths import PROJECT_ROOT as ROOT

POLICY = ROOT / 'manuscript/tables/result_source_policy.json'


def original_results_active():
    return POLICY.exists() and json.loads(POLICY.read_text(encoding='utf-8'))['mode'] == 'original_stopping_results'


def primary_source():
    return ROOT / ('results/paper_original_reporting/primary' if original_results_active()
                   else 'results/hcp181_merged/primary')


def boundary_source():
    return ROOT / ('results/paper_original_reporting/boundary' if original_results_active()
                   else 'results/hcp181_boundary_b122/precision/analysis')
