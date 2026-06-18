# JSON output schema

Official schema for the educational ARVI-RX prototype.

```json
{
  "image_quality": "good | limited | poor",
  "predicted_class": "normal | suspected_opacity | uncertain",
  "confidence": 0.0,
  "visual_evidence": ["short visible observation"],
  "justification": "string",
  "limitations": ["string"],
  "warning": "Prototype pédagogique. Non destiné au diagnostic. Validation par un professionnel qualifié requise."
}
```

## Official decisions

- The official visual evidence key is `visual_evidence`.
- Do not use `visual_observations` in the final contract.
- `predicted_class` must be one of: `normal`, `suspected_opacity`, `uncertain`.
- `confidence` must be numeric between 0 and 1.
- `warning` is mandatory in 100% of outputs.
- The `uncertain` class is a safety feature and must not be removed.
- The output is educational and non-clinical; it must not contain a definitive diagnosis.

## Interpretation limits

A valid JSON output proves that the software contract is respected. It does not prove medical validity, clinical safety, or diagnostic performance.
