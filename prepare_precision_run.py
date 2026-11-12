"""Prepare the existing paper-specific precision workflow, without fitting."""
import argparse,json,shutil
from pathlib import Path

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",type=Path,default=Path(__file__).resolve().parent)
    ap.add_argument("--run",type=Path,required=True)
    args=ap.parse_args()
    root=args.root.resolve()
    source=args.run if args.run.is_absolute() else root/args.run
    out=root/"results/hcp181_primary_refinement_b101"
    if (out/"host_status.json").exists() or (out/"tasks").exists():
        raise RuntimeError("Existing precision outputs retained; do not prepare over them")
    (out/"inputs").mkdir(parents=True,exist_ok=True)
    for name in ["config.yaml","outer_results.csv"]:
        shutil.copy2(source/name,out/"inputs"/name)
    shutil.copy2(source/"units/pooled_predictions.csv",out/"inputs/pooled_predictions.csv")
    shutil.copy2(root/"code/configs/primary_refinement_manifest.json",out/"manifest.json")
    runtime=out/"runtime/code"
    shutil.copytree(root/"code/src",runtime/"src",dirs_exist_ok=True)
    (runtime/"scripts").mkdir(parents=True,exist_ok=True)
    for name in ["_bootstrap.py","03_run_decoding.py"]:
        shutil.copy2(root/"code/scripts"/name,runtime/"scripts"/name)
    print(out)

if __name__=="__main__":main()
