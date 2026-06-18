# Résumé des éléments existants réutilisés

## Fichiers existants réutilisés

- `README.md` : cadrage pédagogique, warning et commandes.
- `src/inference.py` : baseline déterministe `toy_predict`.
- `src/guardrails.py` : validation JSON, warning et classes autorisées.
- `src/preprocessing.py` : `load_image` et `basic_quality_flag`.
- `src/database.py` et `sql/schema.sql` : persistance SQLite existante.
- `src/metrics.py` : métriques de base déjà présentes.
- `api/main.py` : endpoint `/predict` existant.
- `app/streamlit_app.py` : démo upload existante.
- `app/gradio_app.py` : démo optionnelle existante.
- `eval/run_evaluation.py` : évaluation avec exports et logs SQLite.
- `tests/test_repository_smoke.py` : tests de contrat et smoke test.
- `data/synthetic_cases.csv` et `data/sample_images/` : 30 cas synthétiques.
- `prompts/` et `docs/` : base documentaire existante.

## Éléments manquants ou à consolider

- `src/pipeline.py` n'existe pas encore.
- Les logs SQLite sont écrits pendant l'évaluation, mais pas garantis depuis API/Streamlit.
- La consigne mentionne `visual_observations`, le code actuel utilise `visual_evidence`.
- Le dataset utilise `image_path` et `label`, pas `filename` et `expected_label`.
- Les métriques avancées doivent être transférées dans `src/metrics.py`.
- L'analyse d'erreurs 20-30 cas doit être relue et enrichie par l'équipe.
