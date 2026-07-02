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
- `remote_medgemma`: local Streamlit sends one image to a Colab/ngrok FastAPI
  endpoint where MedGemma runs on GPU, then displays the structured JSON.

The stabilized demo path is `remote_medgemma`. The local machine does not need
to load MedGemma weights; it only sends the image to the remote API and displays
the returned JSON. `mock_medgemma` remains available as a local UI fallback when
Colab is not running.

The real `medgemma` mode may require:

- Hugging Face account and model access approval.
- A valid token in your local environment.
- Enough RAM/VRAM to load a 4B vision-language model.
- Installed runtime dependencies from `requirements.txt`.

If the model cannot be loaded or the response is malformed, the pipeline returns
`uncertain` with a clear limitation and keeps the non-clinical warning.

For the main demo on a local PC with limited RAM/VRAM, use `remote_medgemma`
rather than loading MedGemma locally. The local app sends:

- the uploaded or selected RSNA test image as multipart field `file`;
- the strict ARVI-RX prompt as form field `prompt`;
- the request to `<COLAB_OR_NGROK_URL>/predict`.

The prompt strategy and the retained `remote_medgemma_colab_v1` prompt are
documented in [`docs/prompts.md`](prompts.md).

The Colab API should expose:

- `GET /health` returning `{"status": "ok", ...}`;
- `POST /predict` returning JSON with `class`, `confidence`,
  `observations`, `justification`, `limits`, and `warning`.

The local client also accepts compatible internal keys
`predicted_class`, `visual_evidence`, and `limitations`, then normalizes them
to the repository schema. Text around the JSON is parsed when possible; missing
fields, timeout, unavailable API, or non-JSON responses become an `uncertain`
result with a clear `error_detail`.

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

Remote MedGemma smoke test on a few held-out RSNA cases:

```powershell
python eval/run_evaluation.py --mode remote_medgemma --limit 3 --remote-url https://xxxx.ngrok-free.app --out-dir outputs --db-path outputs/assistant_radio.sqlite
```

Balanced technical evaluation with remote MedGemma:

```powershell
python eval/run_evaluation.py --mode remote_medgemma --per-class-limit 3 --remote-url https://fragile-suing-goldmine.ngrok-free.dev --out-dir outputs_remote_medgemma_balanced_6 --db-path outputs_remote_medgemma_balanced_6/assistant_radio.sqlite
```

For MedGemma modes, the evaluation writes generic files such as
`predictions.csv`, `metrics.json`, `error_analysis.csv`, and, when matplotlib is
available, `confusion_matrix.png`.

## Balanced technical evaluation

The validated remote demo was checked on a small balanced RSNA test subset:

- 3 cases labelled `normal`;
- 3 cases labelled `suspected_opacity`;
- total `n = 6`;
- `json_valid_rate = 1.0`;
- `warning_rate = 1.0`;
- median latency around `22660.5 ms`.

Interpretation: `technical validation only; not medical performance`.

This run only verifies that the end-to-end path works:

```text
RSNA image -> Streamlit/local evaluation -> Colab/ngrok API -> MedGemma -> JSON -> local display/metrics
```

It must not be interpreted as evidence of clinical accuracy. The sample is too
small and the labels are dataset labels, not a clinical validation protocol.

## Remote MedGemma commands

Start the local Streamlit interface:

```powershell
python -m streamlit run app/streamlit_app.py
```

Check the remote Colab/ngrok API. Depending on the notebook version, the health
endpoint may be `/health` or `/`:

```powershell
curl.exe -i -H "ngrok-skip-browser-warning: true" https://fragile-suing-goldmine.ngrok-free.dev/health
curl.exe -i -H "ngrok-skip-browser-warning: true" https://fragile-suing-goldmine.ngrok-free.dev/
```

Test remote prediction with one RSNA image:

```powershell
curl.exe -X POST "https://fragile-suing-goldmine.ngrok-free.dev/predict" `
  -H "ngrok-skip-browser-warning: true" `
  -F "file=@data\kaggle_raw\rsna_pneumonia\Training\Images\e5dbda69-4ffc-4bbe-bec8-95a7d08f80b6.png"
```

Run the balanced technical evaluation:

```powershell
python eval/run_evaluation.py `
  --mode remote_medgemma `
  --per-class-limit 3 `
  --remote-url "https://fragile-suing-goldmine.ngrok-free.dev" `
  --out-dir outputs_remote_medgemma_balanced_6 `
  --db-path outputs_remote_medgemma_balanced_6\assistant_radio.sqlite
```

## Difficulty encountered and correction

During integration, Streamlit initially reported that the remote MedGemma API was
unavailable even though the same ngrok URL worked with `curl.exe`.

Cause: local `HTTP_PROXY` / `HTTPS_PROXY` environment variables were picked up by
Python `requests`, so Streamlit/Python attempted to route requests through an
invalid proxy while `curl.exe` succeeded.

Correction: the remote client uses a dedicated `requests.Session` with
`trust_env = False`, so calls to the Colab/ngrok API ignore those proxy
environment variables. The client also sends the ngrok header
`ngrok-skip-browser-warning: true`.

## Web demo

```powershell
streamlit run app/streamlit_app.py
```

The UI lets you choose `toy`, `baseline`, `improved`, `mock_medgemma`, or
`medgemma`/`remote_medgemma`. The displayed JSON remains experimental and must
include the non-clinical warning. Use `mock_medgemma` as the local fallback when
Colab is not available.

## Limits

- The toy baseline reads synthetic signal from filenames, so it is not medical
  inference.
- The balanced remote MedGemma evaluation uses a very small sample and is only a
  technical integration check.
- The demo depends on Colab/ngrok availability and the current public URL.
- Latency is high because each image is sent to a remote GPU runtime and then
  generated by a 4B vision-language model.
- MedGemma output is parsed and constrained, but this does not validate clinical
  safety or performance.
- Metrics are technical checks on a chosen dataset and label mapping.
- Any real dataset source, licence, access condition, and redistribution
  restriction must be documented in the final report.
- RSNA is used here for tests and evaluation splits, not for a completed
  clinical validation.
- No full MedGemma training has been performed in this repository.
- No fine-tuning has been performed for the validated remote demo.
- LoRA/QLoRA fine-tuning remains future work and must stay clearly separated
  from the stabilized demo path.
- If a future LoRA/QLoRA experiment is attempted, it should use a small
  educational subset, for example 100 to 500 images, with text targets derived
  from available labels only. It must not invent real radiology reports and must
  not be presented as clinically validated.
