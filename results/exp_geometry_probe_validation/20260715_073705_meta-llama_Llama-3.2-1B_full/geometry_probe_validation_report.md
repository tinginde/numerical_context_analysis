# Geometry and Probe Validation Report

This report summarizes a descriptive geometry-and-probing validation study. It does not use activation patching, logit lens, attention-head attribution, MLP attribution, or normalization ablations.

## Configuration

```json
{
  "model": "meta-llama/Llama-3.2-1B",
  "output_root": "results/exp_geometry_probe_validation",
  "values_start": 40,
  "values_stop": 180,
  "values_step": 10,
  "batch_size": 4,
  "seed": 13,
  "ridge_alpha": 10.0,
  "bootstrap_samples": 500,
  "max_pairs_per_category": 20000,
  "smoke_test": false,
  "save_activations": false,
  "device": "cuda",
  "validate_load_only": false,
  "model_id": "meta-llama/Llama-3.2-1B",
  "resolved_revision": "4e20de362430cd3b72f300e6b0f18e50e7166e08",
  "parameter_count": 1235814400,
  "dtype": "bfloat16",
  "transformers_version": "4.57.1"
}
```

## What Was Measured

- Residual-stream hidden states were extracted at every model layer.
- Two positions were analyzed: final numeral token and prompt-final token.
- Pairwise cosine similarity was computed as raw, layer-wise mean-centered, and layer-wise per-dimension standardized cosine.
- Ridge probes predicted z-scored numeric value under held-out-template and held-out-interpolation-value splits.

## Pair Balance Check

- same_context_different_number: n=7875, abs_delta mean=53.33, median=50.00, range=[10, 140].
- same_number_different_context: n=1125, abs_delta mean=0.00, median=0.00, range=[0, 0].
- same_context_same_number_different_template: n=450, abs_delta mean=0.00, median=0.00, range=[0, 0].
- random_baseline: n=15750, abs_delta mean=53.33, median=50.00, range=[10, 140].

## Probe Highlights

- numeral_final, held_out_templates: best true-label R2=1.000 at layer 0 with MAE=0.000.
- numeral_final, held_out_interpolation_values: best true-label R2=0.978 at layer 5 with MAE=0.104.
- prompt_final, held_out_templates: best true-label R2=0.000 at layer 0 with MAE=0.864.
- prompt_final, held_out_interpolation_values: best true-label R2=0.938 at layer 6 with MAE=0.170.

## Interpretation Guardrails

- High raw cosine alone may reflect anisotropy or shared-template effects.
- High corrected cosine does not by itself show that numerical information is absent.
- Above-control probe performance means magnitude is linearly decodable; it does not show causal use.
- A reduction in probe performance is not proof that the model has forgotten the number.
- This experiment does not test task-dependent construction, because there is no decision task or causal intervention.

## Output Files

- `stimuli.csv`: generated prompts with validated numeral spans.
- `pairwise_measurements.csv`: sampled pairwise cosine rows.
- `pairwise_summary.csv`: compact layer/category summaries.
- `pairwise_regression.csv`: descriptive regressions of similarity on same-context, same-number, and absolute numeric distance.
- `probe_results.csv`: ridge probe metrics and controls.
- `*.png`: layer-wise geometry and probe plots.