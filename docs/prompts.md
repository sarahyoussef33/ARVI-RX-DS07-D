# Prompts ARVI-RX / RadioX

Ce document trace les prompts utilises ou retenus dans le prototype ARVI-RX.
Le projet reste un prototype pedagogique non clinique. Les prompts servent a
obtenir une sortie structuree, prudente et auditable; ils ne valident pas une
performance medicale.

## P0 - Prompt baseline simple

**Objectif.** Obtenir une premiere sortie exploitable pour une radiographie
thoracique frontale dans le cadre educatif du projet.

**Extrait representatif.**

```text
You are an educational radiology assistant used in an engineering project.
This is a non-clinical prototype. You must not diagnose, triage,
recommend treatment, or replace a qualified professional.

Choose exactly one predicted_class among:
- normal
- suspected_opacity
- uncertain

Return only valid JSON with exactly these core keys:
image_quality, predicted_class, confidence, visual_evidence,
justification, limitations, warning.
```

**Format de sortie attendu.**

```json
{
  "image_quality": "good | limited | poor",
  "predicted_class": "normal | suspected_opacity | uncertain",
  "confidence": 0.0,
  "visual_evidence": ["short visible observation"],
  "justification": "2 to 4 cautious sentences linked only to visible evidence",
  "limitations": ["synthetic or limited context", "no clinical validation"],
  "warning": "Prototype pedagogique. Non destine au diagnostic. Validation par un professionnel qualifie requise."
}
```

**Contraintes de prudence.**

- Utiliser `uncertain` si l'image est ambigue, de faible qualite ou si les
  indices visuels sont faibles.
- Ne pas inventer d'historique patient, de symptomes ou de contexte clinique.
- Ne pas formuler de diagnostic definitif.
- Conserver le warning medical.

**Limites.**

Le prompt demande deja un JSON, mais il reste assez general. Il peut laisser le
modele produire une justification trop libre ou une confiance trop optimiste si
le modele n'est pas strictement contraint.

**Risques observes ou anticipes.**

- Texte libre autour du JSON selon le modele utilise.
- Confusion entre observation visuelle et interpretation clinique.
- Classe `suspected_opacity` formulee comme diagnostic si le modele n'est pas
  suffisamment prudent.

## P1 - Prompt structure JSON

**Objectif.** Renforcer le contrat de sortie et introduire une strategie de
prudence plus explicite.

**Extrait representatif.**

```text
Use a cautious educational strategy.
1. First assess image_quality: good, limited, or poor.
2. Identify only visible, supportable visual_evidence.
3. Before predicting suspected_opacity, consider poor exposure, rotation,
   low inspiration, projection, overlap, or synthetic artefacts.
4. Prefer "uncertain" when evidence is weak, image_quality is limited/poor,
   observations are vague, or confidence would be below 0.60.
5. Keep confidence conservative and never present it as calibrated clinical probability.

Return only valid JSON with:
image_quality, predicted_class, confidence, visual_evidence,
justification, limitations, warning.
```

**Format de sortie attendu.**

Identique au schema interne ARVI-RX:

```json
{
  "image_quality": "good | limited | poor",
  "predicted_class": "normal | suspected_opacity | uncertain",
  "confidence": 0.0,
  "visual_evidence": ["short factual observation"],
  "justification": "2 to 4 cautious sentences tied to visible evidence",
  "limitations": ["no clinical context", "not a validated medical model"],
  "warning": "Prototype pedagogique. Non destine au diagnostic. Validation par un professionnel qualifie requise."
}
```

**Contraintes de prudence.**

- Preferer `uncertain` sous 0.60 de confiance.
- Forcer `uncertain` si la qualite image est `limited` ou `poor`.
- Limiter les observations a ce qui est visible.
- Interdire les faits medicaux hallucines, l'historique patient et le diagnostic
  definitif.

**Limites.**

Ce prompt est plus defendable pour un rapport de projet, mais il depend encore
du respect du format par le modele. Les garde-fous du code restent necessaires.

**Risques observes ou anticipes.**

- JSON invalide si le modele ajoute du texte avant/apres.
- Sur-utilisation de `uncertain` sur des images difficiles.
- Apparence de precision excessive si la confiance est mal interpretee.

## P2 - Prompt renforce prudent remote_medgemma_colab_v1

**Objectif.** Prompt final retenu pour `remote_medgemma`: faire une inference
multimodale distante via Colab/ngrok et obliger MedGemma a retourner un JSON
court, prudent et exploitable par RadioX.

**Version tracee.**

```text
PROMPT_VERSION = "remote_medgemma_colab_v1"
```

**Texte du prompt utilise dans le notebook Colab.**

```text
Tu es un assistant radiologue virtuel pedagogique pour le projet ARVI-RX.
Analyse cette radiographie thoracique frontale.
Reponds uniquement avec un JSON valide. N'ajoute aucun texte hors JSON.

Schema obligatoire :
{
  "class": "normal" | "pneumonia_suspected" | "uncertain",
  "confidence": nombre entre 0 et 1,
  "observations": ["observation courte et prudente"],
  "justification": "justification courte et prudente",
  "limits": "limites de l analyse",
  "warning": "message rappelant que ce n est pas un avis medical"
}

Contraintes :
- Ne donne pas de diagnostic medical definitif.
- Si l'image est ambigue, de mauvaise qualite ou si tu n'es pas sur, utilise "uncertain".
- Les seules classes autorisees sont normal, pneumonia_suspected, uncertain.
```

**Format de sortie attendu.**

Le notebook Colab retourne le contrat remote:

```json
{
  "class": "normal | pneumonia_suspected | uncertain",
  "confidence": 0.0,
  "observations": ["observation courte et prudente"],
  "justification": "justification courte et prudente",
  "limits": "limites de l analyse",
  "warning": "message rappelant que ce n est pas un avis medical"
}
```

Le client local normalise ensuite ce format vers le schema interne ARVI-RX:

```json
{
  "predicted_class": "normal | suspected_opacity | uncertain",
  "confidence": 0.0,
  "visual_evidence": ["observation courte et prudente"],
  "justification": "justification courte et prudente",
  "limitations": ["limites de l analyse"],
  "warning": "Prototype pedagogique. Non destine au diagnostic. Validation par un professionnel qualifie requise.",
  "prompt_version": "remote_medgemma_colab_v1"
}
```

**Contraintes de prudence.**

- JSON valide uniquement, sans texte avant ni apres.
- Trois classes autorisees: `normal`, `pneumonia_suspected`, `uncertain`.
- Confiance numerique entre 0 et 1.
- Observations visuelles courtes et prudentes.
- Justification courte.
- Limites explicites.
- Avertissement medical.
- Aucune conclusion diagnostique definitive.

**Limites.**

La sortie reste produite par un modele generatif. Meme si le prompt impose un
JSON, le code doit toujours parser, normaliser et valider la reponse. Le prompt
ne remplace pas une validation clinique.

**Risques observes ou anticipes.**

- Le modele peut retourner du texte autour du JSON; le parseur tente alors
  d'extraire le premier objet JSON.
- Le modele peut utiliser `pneumonia_suspected`, que le client local mappe vers
  `suspected_opacity`.
- Le modele peut fournir une confiance elevee sur un petit echantillon; cela
  reste une verification technique, pas une mesure de performance medicale.
- La latence depend de Colab/ngrok et du GPU disponible.

## Comparaison des prompts

| Prompt | Sortie structuree | Warning medical | Gestion de l'incertitude | Risque de texte libre | Interet pour le projet |
|---|---|---|---|---|---|
| P0 baseline simple | Oui, demande JSON | Oui | Basique | Moyen | Point de depart lisible pour expliquer le contrat minimal |
| P1 structure JSON prudent | Oui, schema interne ARVI-RX | Oui | Forte, seuil 0.60 et qualite image | Moyen a faible | Bon support pour expliquer les garde-fous et le mode improved |
| P2 remote_medgemma_colab_v1 | Oui, JSON strict remote | Oui, force ensuite par le pipeline local | Forte, classe `uncertain` demandee si ambigu | Faible mais non nul | Prompt final retenu pour la demo remote MedGemma |

## Prompt final retenu

Le prompt final retenu pour la demonstration `remote_medgemma` est P2, car il
impose:

- un JSON valide;
- les trois classes autorisees cote remote: `normal`, `pneumonia_suspected`,
  `uncertain`;
- le mapping local vers les classes ARVI-RX: `normal`, `suspected_opacity`,
  `uncertain`;
- une confiance entre 0 et 1;
- des observations visuelles prudentes;
- une justification courte;
- des limites;
- un avertissement medical;
- aucune conclusion diagnostique definitive.

Cette decision est coherente avec l'objectif du projet: prioriser une inference
prudente, structuree, tracable et evaluable, sans pretendre a une validation
clinique.

## Chatbot RadioX

Le chatbot de l'interface RadioX n'est pas un modele medical. Par defaut, il
n'appelle aucune API externe et ne fait pas de diagnostic.

Son role est limite:

- lire `st.session_state.last_prediction_json`;
- expliquer la classe, la confiance, les observations, la justification, les
  limites, le warning, la latence et le mode pipeline si ces champs existent;
- refuser les demandes de diagnostic, de traitement ou de conduite medicale;
- rappeler que le systeme est un prototype pedagogique;
- conseiller une validation par un professionnel qualifie.

Le chatbot ne doit pas inventer d'information absente du JSON. Il sert a rendre
la sortie plus comprehensible pour la demonstration, pas a interpreter
medicalement une radiographie.

## Pourquoi pas de fine-tuning

Aucun fine-tuning complet n'a ete realise dans ce projet.

Raisons:

- ressources locales limitees pour entrainer ou adapter un modele
  vision-langage de plusieurs milliards de parametres;
- absence de protocole de validation clinique;
- donnees disponibles utilisees pour tests, demonstration et evaluation
  technique, pas pour produire un modele medical valide;
- priorite donnee a l'inference distante prudente, au JSON, aux garde-fous, a la
  tracabilite et a l'analyse d'erreurs;
- risque de presenter a tort une adaptation comme une amelioration clinique.

LoRA/QLoRA reste une piste future possible, separee de la demo stabilisee. Si
elle est exploree, elle doit rester experimentale, utiliser un petit sous-
ensemble educatif, documenter ses limites et ne jamais etre presentee comme une
validation clinique.
