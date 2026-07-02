from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.guardrails import validate_prediction
from src.metrics import confusion_matrix, per_class_metrics, specificity_metrics, summarize_metrics
from src.database import init_db
from src.pipeline import run_pipeline


EXPECTED_REAL_CLASSES = ["normal", "suspected_opacity"]


def read_cases(path: Path) -> list[dict]:
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def resolve_cases(mode: str, metadata_path: Path | None = None, use_synthetic: bool = False) -> list[dict]:
    if use_synthetic:
        return read_cases(ROOT / 'data' / 'synthetic_cases.csv')

    candidates = []
    if metadata_path:
        candidates.append(metadata_path)
    candidates.extend([ROOT / 'data' / 'splits' / 'test.csv', ROOT / 'data' / 'metadata.csv'])

    for path in candidates:
        if path.exists():
            rows = read_cases(path)
            if rows:
                return rows

    if mode in {'baseline', 'improved', 'mock_medgemma', 'remote_medgemma'}:
        return read_cases(ROOT / 'data' / 'synthetic_cases.csv')

    raise SystemExit(
        'No real dataset rows found. Run data/prepare_real_dataset.py first, '
        'or use --mode mock_medgemma for an offline pipeline check.'
    )


def select_balanced_cases(
    cases: list[dict],
    per_class_limit: int,
    expected_classes: list[str] = EXPECTED_REAL_CLASSES,
) -> list[dict]:
    if per_class_limit <= 0:
        raise SystemExit("--per-class-limit must be a positive integer.")

    counts = Counter(row.get("label", "") for row in cases)
    missing = [label for label in expected_classes if counts.get(label, 0) == 0]
    if missing:
        available = ", ".join(f"{label}={count}" for label, count in sorted(counts.items()))
        raise SystemExit(
            "Cannot build a balanced evaluation sample because these classes are missing "
            f"from the loaded cases: {missing}. Available labels: {available}"
        )

    selected_by_label: dict[str, list[dict]] = defaultdict(list)
    for case in cases:
        label = case.get("label", "")
        if label in expected_classes and len(selected_by_label[label]) < per_class_limit:
            selected_by_label[label].append(case)
        if all(len(selected_by_label[label]) >= per_class_limit for label in expected_classes):
            break

    insufficient = [
        f"{label}: requested {per_class_limit}, available {len(selected_by_label[label])}"
        for label in expected_classes
        if len(selected_by_label[label]) < per_class_limit
    ]
    if insufficient:
        raise SystemExit("Not enough cases for balanced evaluation: " + "; ".join(insufficient))

    balanced: list[dict] = []
    for index in range(per_class_limit):
        for label in expected_classes:
            balanced.append(selected_by_label[label][index])
    return balanced


def build_error_analysis(rows: list[dict]) -> list[dict]:
    error_rows = []
    for row in rows:
        is_correct = row['label'] == row['predicted_class']
        if is_correct and row['predicted_class'] != 'uncertain':
            continue
        if row['predicted_class'] == 'uncertain' and row['label'] != 'uncertain':
            error_type = 'overcautious_uncertain'
        elif row['label'] == 'uncertain' and row['predicted_class'] != 'uncertain':
            error_type = 'missed_uncertainty'
        elif not is_correct:
            error_type = 'wrong_class'
        else:
            error_type = 'review_uncertain'
        error_rows.append({
            'case_id': row.get('case_id', Path(row['filename']).stem),
            'filename': row['filename'],
            'expected_label': row['label'],
            'predicted_class': row['predicted_class'],
            'confidence': row['confidence'],
            'error_type': error_type,
            'error_detail': row.get('error_detail', ''),
            'justification': row.get('justification', ''),
            'review_comment': 'technical error analysis only; not clinical review',
        })
    return error_rows


def write_confusion_png(path: Path, matrix_rows: list[dict]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    labels = [row['expected_label'] for row in matrix_rows]
    values = [[row[label] for label in labels] for row in matrix_rows]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(values, cmap='Blues')
    ax.set_xticks(range(len(labels)), labels=labels, rotation=30, ha='right')
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Expected')
    for i, row in enumerate(values):
        for j, value in enumerate(row):
            ax.text(j, i, str(value), ha='center', va='center')
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def run(
    mode: str,
    db_path: Path,
    metadata_path: Path | None = None,
    limit: int | None = None,
    per_class_limit: int | None = None,
    use_synthetic: bool = False,
    remote_url: str | None = None,
) -> tuple[list[dict], dict]:
    cases = resolve_cases(mode, metadata_path, use_synthetic=use_synthetic)
    if per_class_limit is not None:
        cases = select_balanced_cases(cases, per_class_limit)
    elif limit is not None:
        cases = cases[:limit]
    rows = []
    init_db(db_path)
    for case in cases:
        image_path = ROOT / case['image_path']
        pred = run_pipeline(image_path, mode=mode, db_path=db_path, remote_url=remote_url)
        valid, errors = validate_prediction(pred)
        row = {
            'case_id': case.get('case_id') or Path(case['image_path']).stem,
            'source': case.get('source', ''),
            'label': case['label'],
            'expected_label': case['label'],
            'filename': image_path.name,
            'predicted_class': pred['predicted_class'],
            'confidence': pred['confidence'],
            'json_valid': valid,
            'warning': pred.get('warning', ''),
            'latency_ms': pred.get('latency_ms', 0),
            'guardrail_errors': ';'.join(errors),
            'error_detail': pred.get('error_detail', ''),
            'justification': pred.get('justification', ''),
            'interpretation_note': 'technical validation only; not medical performance',
        }
        rows.append(row)
    metrics = summarize_metrics(rows)
    return rows, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mode',
        choices=['toy', 'baseline', 'improved', 'medgemma', 'mock_medgemma', 'remote_medgemma'],
        default='toy',
    )
    parser.add_argument('--out-dir', type=Path, default=ROOT / 'eval' / 'outputs')
    parser.add_argument('--db-path', type=Path, default=ROOT / 'medical_ai_evidence.sqlite')
    parser.add_argument('--metadata-path', type=Path, default=None)
    parser.add_argument('--limit', type=int, default=None, help='Evaluate only the first N cases.')
    parser.add_argument(
        '--per-class-limit',
        type=int,
        default=None,
        help='Evaluate N cases per expected class from the loaded split.',
    )
    parser.add_argument('--remote-url', default=None, help='Remote Colab/ngrok API URL for remote_medgemma.')
    args = parser.parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    modes = ['baseline', 'improved'] if args.mode == 'toy' else [args.mode]
    summary = []
    for mode in modes:
        rows, metrics = run(
            mode,
            args.db_path,
            metadata_path=args.metadata_path,
            limit=args.limit,
            per_class_limit=args.per_class_limit,
            use_synthetic=args.mode == 'toy',
            remote_url=args.remote_url,
        )
        write_csv(out_dir / f'{mode}_predictions.csv', rows)
        y_true = [row['label'] for row in rows]
        y_pred = [row['predicted_class'] for row in rows]
        matrix = confusion_matrix(y_true, y_pred)
        write_csv(out_dir / f'{mode}_confusion_matrix.csv', matrix)
        write_csv(out_dir / f'{mode}_per_class_metrics.csv', per_class_metrics(y_true, y_pred))
        write_csv(out_dir / f'{mode}_specificity_metrics.csv', specificity_metrics(y_true, y_pred))
        (out_dir / f'{mode}_metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
        write_csv(out_dir / f'{mode}_error_analysis.csv', build_error_analysis(rows))
        write_confusion_png(out_dir / f'{mode}_confusion_matrix.png', matrix)
        if mode in {'medgemma', 'mock_medgemma', 'remote_medgemma'}:
            write_csv(out_dir / 'predictions.csv', rows)
            (out_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
            write_csv(out_dir / 'error_analysis.csv', build_error_analysis(rows))
            write_csv(out_dir / 'confusion_matrix.csv', matrix)
            write_csv(out_dir / 'per_class_metrics.csv', per_class_metrics(y_true, y_pred))
            write_confusion_png(out_dir / 'confusion_matrix.png', matrix)
        summary.append({'mode': mode, **metrics})
    write_csv(out_dir / 'before_after_summary.csv', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
