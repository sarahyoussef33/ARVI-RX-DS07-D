# Real data and MedGemma 4B

This project remains an educational prototype. It is not a medical device and
must not be used to diagnose, triage, or guide patient care.

## Data layout

Expected real-data structure:

```text
data/
  real_images/
    NORMAL/
    PNEUMONIA/
  metadata.csv
  splits/
    train.csv
    val.csv
    test.csv
```

`metadata.csv` and each split use at least these columns:

```csv
image_path,label,source
```

Do not commit patient-identifying data. Only use public, authorized, and
de-identified data whose licence permits your educational use.

## Label mapping

The preparation script normalizes common raw dataset labels:

| Raw label example | Project label |
|---|---|
| `NORMAL`, `NO_FINDING` | `normal` |
| `PNEUMONIA`, `OPACITY`, `LUNG_OPACITY` | `suspected_opacity` |
| unknown, ambiguous, unsupported labels | `uncertain` |

This mapping is deliberately conservative. It is a project taxonomy, not a
clinical truth standard.

## Prepare real data

From class folders:

```powershell
python data/prepare_real_dataset.py --images-dir data/real_images --source kaggle_chest_xray
```

From an existing CSV containing `image_path,label`:

```powershell
python data/prepare_real_dataset.py --input-csv path/to/source.csv --source dataset_name
```

The script writes:

- `data/metadata.csv`
- `data/splits/train.csv`
- `data/splits/val.csv`
- `data/splits/test.csv`

It verifies that files exist, normalizes labels, creates reproducible splits,
and groups rows by image stem to reduce data leakage risk.

## MedGemma modes

Available pipeline modes:

- `toy`: original deterministic baseline, kept for reproducible comparison.
- `baseline`: toy baseline.
- `improved`: toy baseline with stronger uncertainty behavior.
- `mock_medgemma`: offline MedGemma-shaped output for tests and demos.
- `medgemma`: real Hugging Face inference with `google/medgemma-4b-it`.

The real `medgemma` mode may require:

- Hugging Face account and model access approval.
- A valid token in your local environment.
- Enough RAM/VRAM to load a 4B vision-language model.
- Installed runtime dependencies from `requirements.txt`.

If the model cannot be loaded or the response is malformed, the pipeline returns
`uncertain` with a clear limitation and keeps the non-clinical warning.

## Evaluation

Toy comparison:

```powershell
python eval/run_evaluation.py --mode toy --out-dir outputs --db-path outputs/assistant_radio.sqlite
```

Offline MedGemma-shaped smoke test:

```powershell
python eval/run_evaluation.py --mode mock_medgemma --out-dir outputs --db-path outputs/assistant_radio.sqlite
```

Real MedGemma evaluation after preparing real data:

```powershell
python eval/run_evaluation.py --mode medgemma --out-dir outputs --db-path outputs/assistant_radio.sqlite
```

For MedGemma modes, the evaluation writes generic files such as
`predictions.csv`, `metrics.json`, `error_analysis.csv`, and, when matplotlib is
available, `confusion_matrix.png`.

## Web demo

```powershell
streamlit run app/streamlit_app.py
```

The UI lets you choose `toy`, `baseline`, `improved`, `mock_medgemma`, or
`medgemma`. The displayed JSON remains experimental and must include the
non-clinical warning.

## Limits

- The toy baseline reads synthetic signal from filenames, so it is not medical
  inference.
- MedGemma output is parsed and constrained, but this does not validate clinical
  safety or performance.
- Metrics are technical checks on a chosen dataset and label mapping.
- Any real dataset source, licence, access condition, and redistribution
  restriction must be documented in the final report.
