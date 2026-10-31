"""Compare a fresh Full-method result and this ablation on the same seen140."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
FIELDS = ('method', 'initialization', 'updates', 'split', 'tasks', 'successes', 'success_rate')
sys.path.insert(0,str(ROOT/'src'))
from ablation import contract
SPEC=contract()
ARMS=(('Full RETHINK','Full-Route-SFT-step2000'),
      ('Trigger-only','Trigger-only-SFT-step2000') if SPEC['sft_enabled'] else
      ('External API router','Qwen2.5-3B-Instruct'))


def validate_rows(rows):
    if len(rows) != 2:
        raise ValueError('Expected one Full-method result and one declared ablation result.')
    for row, (name, initialization) in zip(rows, ARMS):
        if (row['method'], row['initialization']) != (name, initialization):
            raise ValueError('Unexpected method or initialization in the result table.')
        if str(row['tasks']) != '140' or str(row['updates']) != '150' or row['split'] != 'valid_seen':
            raise ValueError('Results must use final update 150 and valid_seen with 140 tasks.')
        value = str(row['successes'])
        if not value.isdigit() or not 0 <= int(value) <= 140:
            raise ValueError('Successes must be an integer between 0 and 140.')
        if not math.isclose(float(row['success_rate']), int(value) / 140, rel_tol=0, abs_tol=1e-12):
            raise ValueError('Success rate does not match successes / 140.')


def rows_from_summaries(baseline, method):
    expected_sha = hashlib.sha256((ROOT / 'configs/seen140_tasks.json').read_bytes()).hexdigest()
    rows = []
    for path, (name, initialization) in zip((baseline, method), ARMS):
        result = json.loads(Path(path).read_text(encoding='utf-8'))
        if result.get('status') != 'passed' or result.get('task_manifest_sha256') != expected_sha:
            raise ValueError('Evaluation must be complete and use the pinned seen140 tasks and seeds.')
        if result.get('analyzer_called') is not False or result.get('teacher_called') is not False:
            raise ValueError('Public evaluation must not use an Analyzer or Teacher.')
        if result.get('training_updates') != 0:
            raise ValueError('Evaluation must not update model weights.')
        rows.append(dict(method=name, initialization=initialization, updates=result['step'],
                         split=result['split'], tasks=result['tasks'], successes=result['successes'],
                         success_rate=result['success_rate']))
    validate_rows(rows)
    return rows


def write_table(rows, destination):
    validate_rows(rows)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def render(rows, destination):
    validate_rows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.1), layout='constrained')
    values = [100 * float(row['success_rate']) for row in rows]
    bars = ax.bar([row['method'] for row in rows], values, width=.55, color=['#8c9aaa', '#3979cf'])
    ax.bar_label(bars, labels=[f"{v:.2f}%\n({r['successes']}/140)" for v, r in zip(values, rows)], padding=6)
    ax.set(title='ALFWorld seen140 · Final update 150', ylabel='Task success rate (%)', ylim=(0, 100))
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis='y', alpha=.18)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference','--baseline',dest='baseline',type=Path,help='Full-method output/summary.json')
    parser.add_argument('--ablation','--method',dest='method',type=Path,help='This ablation output/summary.json')
    parser.add_argument('--out', type=Path, default=ROOT / 'results', help='Result directory')
    args = parser.parse_args()
    if (args.baseline is None) != (args.method is None):
        parser.error('--baseline and --method must be provided together.')
    table = args.out / 'tables/seen140.csv'
    try:
        if args.baseline is not None:
            rows = rows_from_summaries(args.baseline, args.method)
        else:
            with table.open(encoding='utf-8', newline='') as stream:
                rows = list(csv.DictReader(stream))
            validate_rows(rows)
        render(rows, args.out / 'figures/seen140.png')
        if args.baseline is not None:
            write_table(rows, table)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
