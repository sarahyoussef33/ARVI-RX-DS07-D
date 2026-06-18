# README notebooks - ARVI-RX

Ces notebooks guident la production du livrable final sans remplacer le code existant.

| Notebook | Responsable recommandé | Rôle | Sorties attendues | À transférer ensuite |
|---|---|---|---|---|
| `00_setup_and_paths.ipynb` | Sarah | Vérifier chemins, dépendances, dossiers et DB temporaire. | `outputs/setup_check.txt` | Chemins robustes et `db_path` configurable. |
| `01_dataset_and_preprocessing.ipynb` | Julia | Vérifier dataset, images, labels et preprocessing. | `outputs/dataset_summary.csv` | Fonctions de preprocessing si retenues. |
| `02_baseline_inference.ipynb` | Hugo | Tester `toy_predict`. | `outputs/baseline_predictions.csv` | Baseline déterministe et contrat clarifié. |
| `03_prompts_and_improved_mode.ipynb` | Julie | Harmoniser prompts et incertitude. | `outputs/improved_predictions.csv`, prompts cibles | Règle d'incertitude. |
| `04_guardrails_and_json_contract.ipynb` | Julie | Tester JSON, warning, classes et cas négatifs. | `outputs/json_contract_check.csv` | Décision `visual_evidence` vs `visual_observations`. |
| `05_pipeline_and_sqlite_logs.ipynb` | Sarah | Préparer `run_pipeline` avec logs SQLite. | `outputs/pipeline_logs_preview.csv` | Futur `src/pipeline.py`. |
| `06_evaluation_and_metrics.ipynb` | Emma | Calculer métriques du livrable. | `outputs/evaluation_predictions.csv`, `outputs/metrics_summary.csv`, `outputs/confusion_matrix.csv`, `outputs/per_class_metrics.csv`, `outputs/specificity_metrics.csv` | Fonctions à ajouter dans `src/metrics.py`. |
| `07_error_analysis_register.ipynb` | Emma | Créer registre commenté de 20 à 30 cas. | `eval/error_analysis.csv` | Registre d'erreurs pour rapport. |
| `08_api_and_web_integration.ipynb` | Sarah + Hugo | Préparer API/Streamlit via pipeline. | Sections de code cible | Modifier API/Streamlit après `src/pipeline.py`. |
| `09_final_smoke_test_and_export.ipynb` | Sarah | Checklist finale et commandes. | `outputs/final_delivery_check.csv` | Smoke test final. |

Ordre recommandé : 00, 01, 02, 03, 04, 05, 06, 07, 08, 09.

Rappels : prototype pédagogique non clinique, warning obligatoire, classe `uncertain` conservée.
