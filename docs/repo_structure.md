# Structure du repo ARVI-RX

Ce document explique l'organisation du depot pour la soutenance. Le projet reste un prototype pedagogique, non clinique.

## Code principal

| Dossier | Role |
|---|---|
| `app/` | Interface Streamlit principale et ancienne interface Gradio optionnelle. |
| `api/` | API FastAPI de demonstration, notamment `/predict`. |
| `src/` | Pipeline logiciel, garde-fous, preprocessing, metriques, SQLite et clients modele. |
| `eval/` | Script d'evaluation, matrices, metriques et registres d'erreurs generiques. |
| `tests/` | Smoke tests et contrats minimaux du repo. |
| `prompts/` | Prompts baseline, improved et schema JSON cible. |
| `sql/` | Schema SQLite utilise pour les logs. |

## Notebooks

Les notebooks officiels sont dans `notebooks/` :

- `00_setup_and_paths.ipynb`
- `01_dataset_and_preprocessing.ipynb`
- `02_baseline_inference.ipynb`
- `03_prompts_and_improved_mode.ipynb`
- `04_guardrails_and_json_contract.ipynb`
- `05_pipeline_and_sqlite_logs.ipynb`
- `06_evaluation_and_metrics.ipynb`
- `07_error_analysis_register.ipynb`
- `08_api_and_web_integration.ipynb`
- `09_final_smoke_test_and_export.ipynb`
- `remote_medgemma_api_colab.ipynb`
- `medgemma_colab_demo.ipynb`

Les notebooks brouillons ou remplaces sont conserves dans `notebooks/archive/`. Ils ne sont pas supprimes afin de garder l'historique pedagogique.

## Donnees

| Chemin | Role |
|---|---|
| `data/metadata.csv` | Catalogue local RSNA prepare, avec `image_path`, `label`, `source`. |
| `data/splits/train.csv` | Split d'apprentissage futur. |
| `data/splits/val.csv` | Split de reglage prompts/seuils. |
| `data/splits/test.csv` | Split d'evaluation technique tenu a part. |
| `data/sample_images/` | Images synthetiques legeres pour smoke tests et demo hors RSNA. |
| `data/kaggle_raw/` | Donnees RSNA locales volumineuses, non versionnees. |

Les donnees RSNA locales ne doivent pas etre modifiees par les nettoyages d'organisation.

## Resultats importants

Le livrable d'evaluation principal est conserve dans :

`outputs_remote_medgemma_20/`

Fichiers importants :

- `predictions.csv`
- `metrics.json`
- `confusion_matrix.csv`
- `per_class_metrics.csv`
- `error_analysis.csv`
- `error_report.md`
- `assistant_radio.sqlite`

Les autres dossiers `outputs_*` correspondent a des essais intermediaires ou controles locaux. Ils peuvent rester localement, mais ils ne constituent pas le livrable principal.

## Rapports

Les documents de soutenance et de cadrage sont dans `docs/`, notamment :

- `docs/final_technical_report.md`
- `docs/architecture.md`
- `docs/evaluation_protocol.md`
- `docs/ethique_et_limites.md`
- `docs/real_data_medgemma.md`
- `docs/prompts.md`

## Lancer Streamlit

Depuis la racine du repo :

```powershell
.\.venv\Scripts\python.exe -m streamlit run app\streamlit_app.py
```

Mode recommande pour la demo :

- `mock_medgemma` si Colab/ngrok n'est pas disponible ;
- `remote_medgemma` si l'API Colab/ngrok est active.

## Lancer l'evaluation

Evaluation locale rapide :

```powershell
.\.venv\Scripts\python.exe eval\run_evaluation.py --mode mock_medgemma --out-dir outputs_check_mock --db-path outputs_check_mock\assistant_radio.sqlite
```

Evaluation remote MedGemma, uniquement si l'URL Colab/ngrok est active :

```powershell
.\.venv\Scripts\python.exe eval\run_evaluation.py --mode remote_medgemma --per-class-limit 10 --out-dir outputs_remote_medgemma_20 --db-path outputs_remote_medgemma_20\assistant_radio.sqlite --remote-url https://xxxx.ngrok-free.dev
```

## Relancer Colab remote_medgemma

1. Ouvrir `notebooks/remote_medgemma_api_colab.ipynb` dans Colab.
2. Demarrer le runtime GPU.
3. Installer/charger les dependances et MedGemma selon le notebook.
4. Lancer l'API exposee par ngrok.
5. Copier l'URL publique ngrok.
6. Coller cette URL dans Streamlit ou dans `--remote-url`.

Le mode `remote_medgemma` depend de Colab, du GPU distant et du reseau. Il est adapte a une demonstration technique limitee, pas a la production.

## Verification avant soutenance

```powershell
python -m compileall -q src api app eval tests
.\.venv\Scripts\python.exe -m pytest -q
```

Statut attendu : compilation OK, tests OK. Les warnings de dependances non bloquants doivent etre mentionnes mais ne remettent pas en cause le prototype.
