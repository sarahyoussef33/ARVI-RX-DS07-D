from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def target_to_label(value: str) -> str:
    return "suspected_opacity" if str(value).strip() == "1" else "normal"


def build_prepared_csv(dataset_dir: Path, output_csv: Path) -> list[dict[str, str]]:
    metadata_path = dataset_dir / "stage2_train_metadata.csv"
    images_dir = dataset_dir / "Training" / "Images"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing train metadata: {metadata_path}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Missing training images directory: {images_dir}")

    by_patient: dict[str, dict[str, str]] = {}
    with metadata_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        required = {"patientId", "Target"}
        if not required <= set(reader.fieldnames or []):
            raise ValueError("stage2_train_metadata.csv must contain patientId and Target columns")

        for row in reader:
            patient_id = row["patientId"]
            image_path = images_dir / f"{patient_id}.png"
            if not image_path.exists():
                continue
            current = by_patient.get(patient_id)
            label = target_to_label(row["Target"])
            if current is None or label == "suspected_opacity":
                by_patient[patient_id] = {
                    "image_path": str(image_path.resolve()),
                    "label": label,
                    "source": "kaggle_rsna_pneumonia",
                }

    rows = sorted(by_patient.values(), key=lambda item: item["image_path"])
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["image_path", "label", "source"])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert the Kaggle RSNA pneumonia processed dataset to ARVI-RX CSV.")
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "data" / "kaggle_raw" / "rsna_pneumonia")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=ROOT / "data" / "kaggle_raw" / "rsna_pneumonia" / "arvi_rsna_metadata.csv",
    )
    args = parser.parse_args()

    rows = build_prepared_csv(args.dataset_dir, args.output_csv)
    counts = Counter(row["label"] for row in rows)
    print(f"prepared_csv: {args.output_csv}")
    print(f"total_images: {len(rows)}")
    for label, count in sorted(counts.items()):
        print(f"{label}: {count}")


if __name__ == "__main__":
    main()
