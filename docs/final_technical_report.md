# Rapport technique final ARVI-RX

## 1. Objectif du projet

ARVI-RX est un prototype p?dagogique d?assistant radiologique virtuel. Il vise ? d?montrer un flux technique complet : chargement d?une radiographie thoracique, appel mod?le, sortie JSON structur?e, garde-fous, journalisation et interface utilisateur compr?hensible.

Le projet ne fournit pas de diagnostic m?dical et ne revendique aucune validation clinique.

## 2. P?rim?tre et non-objectifs

**P?rim?tre couvert** : pipeline logiciel, dataset RSNA pr?par?, splits, modes de pr?diction, contrat JSON, garde-fous, logs SQLite, ?valuation technique limit?e et interface Streamlit.

**Non-objectifs** : diagnostic patient, d?cision clinique, d?ploiement hospitalier, calibration m?dicale des scores, fine-tuning imm?diat.

## 3. Dataset et splits

Le fichier `data/metadata.csv` contient **26684 images** issues de l?int?gration RSNA.

| Classe | Nombre |
| --- | --- |
| normal | 20672 |
| suspected_opacity | 6012 |

Les splits sont mat?rialis?s dans `data/splits/` :

| Split | Total | normal | suspected_opacity | R?le |
| --- | --- | --- | --- | --- |
| train | 18678 | 14480 | 4198 | apprentissage futur |
| val | 4002 | 3117 | 885 | r?glages / choix de seuils |
| test | 4004 | 3075 | 929 | ?valuation finale tenue ? part |

Le split `train` sert ? un futur apprentissage ou fine-tuning. Le split `val` sert aux r?glages de prompts, seuils ou politiques d?incertitude. Le split `test` doit rester tenu ? part pour l??valuation finale technique.

## 4. Architecture

L?architecture suit le flux suivant :

1. Streamlit re?oit une image ou s?lectionne un cas RSNA.
2. `src.pipeline.run_pipeline` charge et valide l?image.
3. Le mode choisi appelle baseline, improved, mock, MedGemma local ou remote MedGemma.
4. Le r?sultat est normalis? en JSON.
5. Les garde-fous valident la classe, la confiance, le warning et le format.
6. Le run est journalis? dans SQLite.
7. L?UI affiche un r?sum? lisible et garde le JSON brut dans un expander technique.

## 5. Modes mod?le

| Mode | R?le | Avantage | Limite | Usage projet |
| --- | --- | --- | --- | --- |
| baseline | R?gles simples / pipeline initial | Rapide, reproductible | Non m?dical, tr?s limit? | Point de comparaison technique |
| improved | Baseline plus prudente | Introduit l?incertitude | Toujours heuristique | Comparer l?effet des garde-fous |
| mock_medgemma | Simulation locale du contrat MedGemma | Fonctionne sans GPU ni r?seau | Ne mesure pas MedGemma | D?mo robuste et tests hors ligne |
| medgemma | Inf?rence locale MedGemma | Cible th?orique autonome | Non viable sur le PC projet | Document? mais non retenu pour la soutenance |
| remote_medgemma | Inf?rence MedGemma via Colab/ngrok | Permet d?utiliser GPU distant | Latence et d?pendance r?seau | Mode principal de d?monstration technique |

## 6. Contrat JSON

Chaque pr?diction doit contenir :

- `image_quality`
- `predicted_class`
- `confidence`
- `visual_evidence`
- `justification`
- `limitations`
- `warning`

Des champs compl?mentaires peuvent exister : `latency_ms`, `pipeline_mode`, `model_name`, `prompt_version`, `db_path`, `remote_url`, `remote_endpoint`.

## 7. Garde-fous

Les garde-fous sont d?finis dans `src/guardrails.py` :

- classes autoris?es : `normal`, `suspected_opacity`, `uncertain` ;
- confiance num?rique entre 0 et 1 ;
- JSON avec cl?s obligatoires ;
- warning obligatoire ;
- fallback vers `uncertain` si le format est invalide ;
- warning r?inject? par le pipeline ;
- validation technique du format avant m?triques.

Ces garde-fous ne garantissent pas la v?rit? clinique. Ils garantissent seulement une sortie exploitable et prudente pour le prototype.

## 8. Logs et SQLite

La structure SQLite est d?crite dans `sql/schema.sql` et utilis?e par `src/database.py`.

Tables principales :

- `cases` : identifiant, image, source, label, split, notes ;
- `prompts` : nom, version et texte de prompt ;
- `runs` : image, mod?le, prompt, JSON de pr?diction, classe, confiance, latence ;
- `evaluations` : label r?el, exactitude, type d?erreur, commentaire.

Les logs sont importants car ils assurent : tra?abilit?, auditabilit?, analyse d?erreurs, comparaison entre modes, et preuve que le warning et le JSON ont bien ?t? produits.

## 9. ?valuation sur 20 images

L??valuation `remote_medgemma` a ?t? r?alis?e sur 20 images du split `test`.

| M?trique | Valeur |
| --- | --- |
| n | 20 |
| accuracy | 0.8 |
| macro_f1 | 0.532 |
| json_valid_rate | 1.0 |
| warning_rate | 1.0 |
| uncertain_rate | 0.0 |
| latency_median_ms | 21976.0 |

Interpr?tation : validation technique encourageante sur un petit ?chantillon, mais non clinique. Le taux JSON et warning ? 1.0 confirme surtout la robustesse du contrat de sortie.

## 10. Analyse d?erreurs

Livrables produits :

- `outputs_remote_medgemma_20/error_analysis.csv`
- `outputs_remote_medgemma_20/confusion_matrix.csv`
- `outputs_remote_medgemma_20/error_report.md`

Matrice :

| Vrai \ Pr?dit | normal | suspected_opacity | uncertain |
| --- | --- | --- | --- |
| normal | 9 | 4 | 0 |
| suspected_opacity | 0 | 7 | 0 |
| uncertain | 0 | 0 | 0 |

Lecture principale : les erreurs observ?es sont surtout des faux positifs, c?est-?-dire des images normales signal?es comme `suspected_opacity`. Le prototype doit donc pr?senter ces sorties comme des signaux ? v?rifier, sans formulation diagnostique.

## 11. GO/NO-GO

| Sujet | D?cision | Justification |
| --- | --- | --- |
| P?rim?tre | GO | Prototype p?dagogique et validation technique clairement d?limit?s. |
| Donn?es | GO | RSNA int?gr? avec metadata et splits; pas de modification des donn?es. |
| Baseline | GO | Disponible pour comparaison technique. |
| remote_medgemma | GO | Fonctionne via Colab/ngrok sur ?chantillon limit?. |
| JSON | GO | Contrat structur? et json_valid_rate=1.0 sur les 20 cas. |
| Warning | GO | Warning obligatoire pr?sent sur tous les outputs test?s. |
| Logs | GO | SQLite journalise les runs et pr?dictions JSON. |
| Latence | NO-GO production | M?diane ~21,976 ms; acceptable pour d?mo, trop lent pour production. |
| Clinique | NO-GO | Aucune validation clinique, aucun diagnostic autoris?. |
| Production | NO-GO | Endpoint ngrok/Colab non stable pour d?ploiement. |
| Fine-tuning | NO-GO maintenant | Piste future seulement; commencer petit et sous supervision. |

## 12. Limites

- ?chantillon remote de 20 images seulement.
- Dataset RSNA binaire simplifi? pour le prototype.
- Pas de revue clinique experte int?gr?e au livrable.
- Confiance non calibr?e m?dicalement.
- Latence ?lev?e via Colab/ngrok.
- D?pendance r?seau et session Colab.
- MedGemma local non viable sur le PC utilis?.
- Pas de fine-tuning r?alis?.

## 13. Am?liorations futures

- Ajouter une revue qualitative syst?matique des erreurs.
- Tester davantage d?images par lots, avec limites de co?t/temps.
- Am?liorer les prompts pour r?duire les faux positifs.
- Ajouter une politique d?incertitude plus stricte.
- Ajouter des visualisations p?dagogiques des r?sultats.
- ?tendre les logs avec versions de prompts et param?tres d?ex?cution.
- Pr?parer un sous-dataset de 5 000 images pour analyses plus robustes.

Le dataset complet contient environ **26 684 images**. Un sous-dataset de **5 000 images** peut ?tre pr?par? pour d?veloppement plus robuste, analyse, s?lection d?exemples, futur classifieur l?ger ou pr?paration fine-tuning. En revanche, ?valuer `remote_medgemma` sur 5 000 images via Colab T4 serait trop long compte tenu d?une latence m?diane d?environ 22 secondes par image.

## 14. Fine-tuning futur LoRA/QLoRA

Le fine-tuning n?est pas r?alis? dans cette soutenance. Il constitue une piste future.

Approche r?aliste :

1. Commencer par 100 ? 500 images, ?quilibr?es et revues.
2. Formaliser les labels et le contrat JSON cible.
3. Utiliser LoRA/QLoRA pour limiter les ressources GPU.
4. Garder un split test jamais vu.
5. Comparer au prompt remote sans fine-tuning.
6. Documenter les erreurs, pas seulement les scores.
7. Ne pas pr?senter le mod?le fine-tun? comme cliniquement valid? sans protocole m?dical adapt?.

Pour cette version, le statut fine-tuning est donc **NO-GO maintenant / GO comme piste future encadr?e**.
