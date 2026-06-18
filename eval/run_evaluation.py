from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.guardrails import validate_prediction
from src.metrics import confusion_matrix, per_class_metrics, specificity_metrics, summarize_metrics
from src.database import init_db
from src.pipeline import run_pipeline


def read_cases(path: Path) -> list[dict]:
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def run(mode: str, db_path: Path) -> tuple[list[dict], dict]:
    cases = read_cases(ROOT / 'data' / 'synthetic_cases.csv')
    rows = []
    init_db(db_path)
    for case in cases:
        image_path = ROOT / case['image_path']
        pred = run_pipeline(image_path, mode=mode, db_path=db_path)
        valid, errors = validate_prediction(pred)
        row = {
            'case_id': case['case_id'],
            'label': case['label'],
            'expected_label': case['label'],
            'filename': image_path.name,
            'predicted_class': pred['predicted_class'],
            'confidence': pred['confidence'],
            'json_valid': valid,
            'warning': pred.get('warning', ''),
            'latency_ms': pred.get('latency_ms', 0),
            'guardrail_errors': ';'.join(errors),
            'interpretation_note': 'technical validation only; not medical performance',
        }
        rows.append(row)
    metrics = summarize_metrics(rows)
    return rows, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['toy', 'baseline', 'improved'], default='toy')
    parser.add_argument('--out-dir', type=Path, default=ROOT / 'eval' / 'outputs')
    parser.add_argument('--db-path', type=Path, default=ROOT / 'medical_ai_evidence.sqlite')
    args = parser.parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    modes = ['baseline', 'improved'] if args.mode == 'toy' else [args.mode]
    summary = []
    for mode in modes:
        rows, metrics = run(mode, args.db_path)
        write_csv(out_dir / f'{mode}_predictions.csv', rows)
        y_true = [row['label'] for row in rows]
        y_pred = [row['predicted_class'] for row in rows]
        write_csv(out_dir / f'{mode}_confusion_matrix.csv', confusion_matrix(y_true, y_pred))
        write_csv(out_dir / f'{mode}_per_class_metrics.csv', per_class_metrics(y_true, y_pred))
        write_csv(out_dir / f'{mode}_specificity_metrics.csv', specificity_metrics(y_true, y_pred))
        (out_dir / f'{mode}_metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
        summary.append({'mode': mode, **metrics})
    write_csv(out_dir / 'before_after_summary.csv', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
