# Geometry and Probe Validation Report

This report summarizes a descriptive geometry-and-probing validation study. It does not use activation patching, logit lens, attention-head attribution, MLP attribution, or normalization ablations.

## Configuration

```json
{
  "model": "hf-internal-testing/tiny-random-LlamaForCausalLM",
  "output_root": "results/exp_geometry_probe_validation",
  "values_start": 40,
  "values_stop": 60,
  "values_step": 10,
  "batch_size": 2,
  "seed": 13,
  "ridge_alpha": 10.0,
  "bootstrap_samples": 50,
  "max_pairs_per_category": 2000,
  "smoke_test": true,
  "save_activations": false,
  "device": "cpu"
}
```

## What Was Measured

- Residual-stream hidden states were extracted at every model layer.
- Two positions were analyzed: final numeral token and prompt-final token.
- Pairwise cosine similarity was computed as raw, layer-wise mean-centered, and layer-wise per-dimension standardized cosine.
- Ridge probes predicted z-scored numeric value under held-out-template and held-out-interpolation-value splits.

## Pair Balance Check

- same_context_different_number: n=36, abs_delta mean=13.33, median=10.00, range=[10, 20].
- same_number_different_context: n=36, abs_delta mean=0.00, median=0.00, range=[0, 0].
- same_context_same_number_different_template: n=9, abs_delta mean=0.00, median=0.00, range=[0, 0].
- random_baseline: n=72, abs_delta mean=13.33, median=10.00, range=[10, 20].

## Probe Highlights

- numeral_final, held_out_templates: best true-label R2=0.000 at layer 0 with MAE=0.816.
- numeral_final, held_out_interpolation_values: best true-label R2=1.000 at layer 0 with MAE=0.000.
- prompt_final, held_out_templates: best true-label R2=0.034 at layer 2 with MAE=0.857.
- prompt_final, held_out_interpolation_values: best true-label R2=1.000 at layer 0 with MAE=0.000.

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