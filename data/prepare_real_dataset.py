from __future__ import annotations

import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ALLOWED_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}
LABEL_MAP = {
    "normal": "normal",
    "no_finding": "normal",
    "no finding": "normal",
    "pneumonia": "suspected_opacity",
    "opacity": "suspected_opacity",
    "lung_opacity": "suspected_opacity",
    "lung opacity": "suspected_opacity",
    "suspected_opacity": "suspected_opacity",
    "uncertain": "uncertain",
    "unknown": "uncertain",
    "ambiguous": "uncertain",
}


def normalize_label(raw_label: str) -> str:
    key = raw_label.strip().lower().replace("-", "_")
    return LABEL_MAP.get(key, "uncertain")


def relpath(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def read_rows_from_csv(input_csv: Path, source: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with input_csv.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if not {"image_path", "label"} <= set(reader.fieldnames or []):
            raise ValueError("Input CSV must contain image_path and label columns")
        for row in reader:
            image_path = Path(row["image_path"])
            if not image_path.is_absolute():
                image_path = (input_csv.parent / image_path).resolve()
            if image_path.suffix.lower() not in ALLOWED_SUFFIXES:
                continue
            if not image_path.exists():
                raise FileNotFoundError(f"Missing image file: {image_path}")
            rows.append(
                {
                    "image_path": relpath(image_path),
                    "label": normalize_label(row["label"]),
                    "source": row.get("source") or source,
                }
            )
    return rows


def read_rows_from_directory(images_dir: Path, source: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for image_path in sorted(images_dir.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in ALLOWED_SUFFIXES:
            continue
        raw_label = image_path.parent.name
        rows.append(
            {
                "image_path": relpath(image_path),
                "label": normalize_label(raw_label),
                "source": source,
            }
        )
    return rows


def split_rows(
    rows: list[dict[str, str]],
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[Path(row["image_path"]).stem].append(row)

    groups = list(grouped.values())
    rng = random.Random(seed)
    rng.shuffle(groups)

    n = len(groups)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    split_groups = {
        "train": groups[:train_end],
        "val": groups[train_end:val_end],
        "test": groups[val_end:],
    }
    return {name: [row for group in split_groups[name] for row in group] for name in split_groups}


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["image_path", "label", "source"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a real CXR dataset for ARVI-RX.")
    parser.add_argument("--images-dir", type=Path, default=ROOT / "data" / "real_images")
    parser.add_argument("--input-csv", type=Path, default=None)
    parser.add_argument("--metadata-out", type=Path, default=ROOT / "data" / "metadata.csv")
    parser.add_argument("--splits-dir", type=Path, default=ROOT / "data" / "splits")
    parser.add_argument("--source", default="real_dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    args = parser.parse_args()

    if args.input_csv:
        rows = read_rows_from_csv(args.input_csv, args.source)
    else:
        rows = read_rows_from_directory(args.images_dir, args.source)

    if not rows:
        raise SystemExit(
            "No real images found. Put images under data/real_images/<RAW_LABEL>/ "
            "or pass --input-csv with image_path,label columns."
        )

    write_csv(args.metadata_out, rows)
    splits = split_rows(rows, seed=args.seed, train_ratio=args.train_ratio, val_ratio=args.val_ratio)
    for name, split_rows_ in splits.items():
        write_csv(args.splits_dir / f"{name}.csv", split_rows_)

    print("Prepared real dataset")
    print(f"metadata: {args.metadata_out}")
    print(f"total: {len(rows)}")
    for label, count in sorted(Counter(row["label"] for row in rows).items()):
        print(f"{label}: {count}")
    for name, split_rows_ in splits.items():
        print(f"{name}: {len(split_rows_)}")


if __name__ == "__main__":
    main()
