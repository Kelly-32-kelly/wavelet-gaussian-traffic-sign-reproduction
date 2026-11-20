"""Sequential, restartable reproduction without inspecting test results for selection."""
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def run(script,*args):
    subprocess.run([sys.executable,'-u',str(ROOT/script),*map(str,args)],check=True)


if __name__=='__main__':
    run('prepare_data.py')
    run('run_features.py')
    for seed in [42,314,2718]:
        run('train_representation.py','--seed',seed)
    if not (ROOT/'results'/'selection'/'locked_configs.json').exists():
        run('evaluate_heads.py','tune')
    for budget in [100,50,25]:
        for seed in [42,314,2718]:
            run('evaluate_heads.py','evaluate','--seed',seed,'--budget',budget)
    run('gap_control.py')
    run('analyse_results.py')
    run('verify_study.py')
