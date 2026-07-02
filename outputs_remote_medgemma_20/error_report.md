# Analyse d'erreurs remote_medgemma sur 20 images

## R?sum?

Cette analyse porte sur un ?chantillon technique de 20 images du split `test` RSNA, ?valu? avec `remote_medgemma` via Colab/ngrok. Les r?sultats ne constituent pas une performance clinique et ne doivent pas ?tre interpr?t?s comme une validation m?dicale.

| Indicateur | Valeur |
| --- | --- |
| Nombre d?images | 20 |
| Accuracy | 0.8 |
| Macro F1 | 0.532 |
| JSON valide | 1.0 |
| Warning pr?sent | 1.0 |
| Taux uncertain | 0.0 |
| Latence m?diane | 21976.0 ms |

## Composition de l??chantillon

| Classe r?elle | Nombre |
| --- | --- |
| normal | 13 |
| suspected_opacity | 7 |

## Matrice de confusion

| Vrai \ Pr?dit | normal | suspected_opacity | uncertain |
| --- | --- | --- | --- |
| normal | 9 | 4 | 0 |
| suspected_opacity | 0 | 7 | 0 |
| uncertain | 0 | 0 | 0 |

## Cat?gories d?erreurs

| Cat?gorie | Nombre | Interpr?tation |
| --- | --- | --- |
| TP | 7 | opacit? correctement signal?e |
| TN | 9 | normal correctement reconnu |
| FP | 4 | radio normale signal?e comme opacit? suspect?e |
| FN | 0 | opacit? non d?tect?e |
| uncertainty_missing | 0 | le mod?le aurait d? rester incertain |
| text_issue | 0 | probl?me de format ou warning |
| none | 0 | cas ? revoir manuellement |

## Exemples de cas corrects

| case_id | vrai | pr?dit | confiance | justification courte |
| --- | --- | --- | --- | --- |
| e5dbda69-4ffc-4bbe-bec8-95a7d08f80b6 | normal | normal | 0.95 | L'image est de bonne qualité et ne montre pas de signes d'opacité ou de cardiomégalie. |
| fc10d91c-63f5-40d5-812c-15e33855d095 | normal | normal | 0.95 | L'image est de bonne qualité et montre une silhouette pulmonaire normale sans anomalies apparentes. |
| e688c2ea-a039-4935-a072-f34e72d1a565 | normal | normal | 0.95 | L'image est de bonne qualité et montre des structures thoraciques normales. |

## Exemples d?erreurs

| case_id | vrai | pr?dit | type | action corrective |
| --- | --- | --- | --- | --- |
| e20af8e8-56f2-486f-962e-7706d77dc5f5 | normal | suspected_opacity | FP | Revoir le prompt et les exemples: r?duire les sur-alertes sur radios normales; demander une justification plus localis?e. |
| ea696000-e0e3-495e-82e1-e0b5ea273cb7 | normal | suspected_opacity | FP | Revoir le prompt et les exemples: r?duire les sur-alertes sur radios normales; demander une justification plus localis?e. |
| 490acf2f-8fc0-419e-b83c-190735b71efb | normal | suspected_opacity | FP | Revoir le prompt et les exemples: r?duire les sur-alertes sur radios normales; demander une justification plus localis?e. |
| db3f619a-81f5-4cd4-b08f-304b8bd9a81a | normal | suspected_opacity | FP | Revoir le prompt et les exemples: r?duire les sur-alertes sur radios normales; demander une justification plus localis?e. |

## Interpr?tation prudente

L??valuation confirme surtout que le flux technique fonctionne : appel distant, parsing JSON, normalisation des champs, garde-fous, warning obligatoire et journalisation SQLite. Le score `accuracy=0.8` sur 20 images est encourageant pour une d?monstration, mais l??chantillon est trop petit pour conclure ? une performance m?dicale.

Le principal signal d?erreur observ? est le faux positif : certaines radios normales sont class?es `suspected_opacity`. Cela montre que le syst?me peut sur-signaler une anomalie. Dans une interface p?dagogique, cette erreur doit ?tre pr?sent?e comme une alerte ? v?rifier, jamais comme un diagnostic.

## Limites

- ?chantillon de 20 images seulement.
- ?valuation effectu?e via un endpoint Colab/ngrok d?pendant du r?seau et de la disponibilit? GPU.
- Latence ?lev?e pour un usage interactif ? grande ?chelle.
- Pas de calibration clinique de la confiance.
- Pas de revue radiologique experte des erreurs dans ce livrable.
- Pas de fine-tuning r?alis?.

## Actions correctives propos?es

- Am?liorer le prompt pour demander des indices visuels localis?s et ?viter la sur-d?tection.
- Ajouter une politique d?incertitude plus prudente lorsque la justification est vague.
- R?aliser une revue qualitative d?un sous-?chantillon d?erreurs avec un encadrant ou expert m?tier.
- Tester un sous-dataset de d?veloppement plus grand avec `mock_medgemma`/baseline pour le flux, puis un ?chantillon remote limit?.
- Pr?parer un futur LoRA/QLoRA sur 100 ? 500 images annot?es et revues, sans l?ex?cuter dans cette soutenance.
