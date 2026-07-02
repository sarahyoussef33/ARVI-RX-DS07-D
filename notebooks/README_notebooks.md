# README notebooks - ARVI-RX

Ces notebooks guident la couche pédagogique du projet avant la création du pipeline final.
Ils ne remplacent pas le code dans `src/` et ne prouvent pas une validité médicale.

## Décisions officielles

- Clé JSON officielle : `visual_evidence`.
- Warning officiel : `Prototype pédagogique. Non destiné au diagnostic. Validation par un professionnel qualifié requise.`
- Dataset source officiel : `image_path` et `label`.
- Les outputs peuvent utiliser `filename` et `expected_label` pour la lisibilité.
- Streamlit est la démo officielle; Gradio reste optionnel.
- `improved` signifie prudence renforcée, pas modèle médical supérieur.
- Accuracy `1.0` = validation technique du pipeline jouet, pas performance clinique.

## Rôle des notebooks

| Notebook | Responsable | Ce qu'il prouve | Ce qu'il ne prouve pas | Sorties |
|---|---|---|---|---|
| `00_setup_and_paths.ipynb` | Sarah | Environnement, chemins, dépendances, DB temporaire | Que l'app finale est intégrée | `outputs/setup_check.txt` |
| `01_dataset_and_preprocessing.ipynb` | Julia | CSV lisible, images présentes, limites dataset connues | Validité clinique des images | `outputs/dataset_summary.csv` |
| `02_baseline_inference.ipynb` | Hugo | Baseline reproductible et contrat JSON testable | Capacité radiologique; le score est biaisé par label leakage | `outputs/baseline_predictions.csv` |
| `03_prompts_and_improved_mode.ipynb` | Julie | Stratégie de prudence documentée | Effet réel d'un VLM, non utilisé ici | `outputs/improved_predictions.csv` |
| `04_guardrails_and_json_contract.ipynb` | Julie | JSON, warning, classes et garde-fous | Sécurité clinique complète | `outputs/json_contract_check.csv` |
| `05_pipeline_and_sqlite_logs.ipynb` | Sarah | Design préparatoire du futur pipeline | Pipeline final déjà codé | `outputs/pipeline_logs_preview.csv` |
| `06_evaluation_and_metrics.ipynb` | Emma | Métriques techniques et exports lisibles | Performance médicale | `outputs/metrics_summary.csv`, matrices et métriques par classe |
| `07_error_analysis_register.ipynb` | Emma | Analyse critique de 20 à 30 cas synthétiques | Analyse d'erreurs clinique | `eval/error_analysis.csv` |
| `08_api_and_web_integration.ipynb` | Sarah + Hugo | Plan d'intégration API/Streamlit | API déjà branchée au pipeline | Code cible documenté |
| `09_final_smoke_test_and_export.ipynb` | Sarah | Checklist de livraison | Qualité scientifique complète | `outputs/final_delivery_check.csv` |

## Archive

Le dossier `notebooks/archive/` conserve des notebooks brouillons ou remplacés par la série officielle ci-dessus.
Ils ne sont pas supprimés afin de garder l'historique du travail, mais ils ne sont pas les supports principaux de soutenance.

## Validation technique vs validation médicale

Les notebooks valident surtout le comportement logiciel: fichiers présents, JSON valide,
warning obligatoire, garde-fous, exports CSV, métriques et registre d'analyse.
Ils ne valident pas un dispositif médical. La baseline lit indirectement les labels via
les noms de fichiers; tout score parfait doit être expliqué comme un résultat attendu du
jeu synthétique.
