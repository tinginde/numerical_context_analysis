# Results

Output figures from all experiments. Subfolders correspond to the experiment table in the project README.

> **Interpretation note:** these are exploratory outputs produced across several stages of the project. Earlier figures should be read together with the later geometry/probe validation, which shows that high within-context cosine does not imply loss of numerical information. See the [project README](../README.md) for the current evidence boundary.

---

## `exp_A_B/` — Cross-Context & Cross-Magnitude (v1)

Per-layer cosine similarity curves and a final-layer heatmap for Experiments A and B.

- **Exp A:** same number `24` across five semantic contexts (flight, study, money, heart rate, apples)
- **Exp B:** same sentence template with values `2.4`, `24`, and `240`

Key output: `numerical_context_analysis_v1.png`

---

## `exp_C/` — Medical Severity Discrimination (v2, v3)

Per-layer cosine similarity and final-layer summaries for LLaMA 3.2-1B (v2) and 3B (v3) across five medical contexts × four severity levels.

- `numerical_context_analysis_v2.png`: Exp C on LLaMA 3.2-1B
- `numerical_context_analysis_v3.png`: Exp C on LLaMA 3.2-3B

These figures test whether dangerous and normal values remain separable within the same medical context.

---

## `exp_D/` — PCA Visualization (v4)

PCA of hidden states for 8 medical contexts × 6 severity levels at embedding, middle, and final layers.

- Point shape = context type
- Point colour = danger level (`0`-`5`)
- Single combined figure compares LLaMA 3.2-1B and 3B

Key output: `numerical_context_analysis_v4.png`

---

## `exp_E/` — Behavioral Evaluation (v5)

Output-level evaluation across 1,152 inferences (`3 formats × 4 prompting strategies × 2 models × 8 contexts × 6 severity values`).

Primary figures:

- `v5_accuracy.png`: overall accuracy by condition
- `v5_consistency.png`: consistency / ordinal agreement by condition
- `v5_risk_alignment.png`: risk-sensitive performance by condition
- `v5_parse_rate.png`: parse success rate by model and prompting condition
- `v5_hybrid_consistency.png`: internal consistency for hybrid-format outputs

Associated analysis files:

- `v5_metrics.csv`: aggregate metrics table
- `v5_raw.csv`: raw inference outputs
- `v5_behavior_summary.csv`: post-hoc behavior summary
- `v5_context_breakdown.csv`: per-context breakdown
- `v5_error_cases.csv`: representative error cases
- `v5_behavior_analysis.md`: narrative analysis
- `v5_behavior_overview.png`: post-hoc overview figure

---

Figures are generated automatically by running the corresponding scripts from the repository root. See the project README for setup and execution instructions.

---

## `exp_geometry_probe_validation/` — Geometry and Linear Decodability

The maintained validation workflow tests whether the earlier context-dominant ordering survives anisotropy correction and whether numerical magnitude remains linearly decodable.

The completed full run contains:

- raw, mean-centered, and per-dimension standardized cosine summaries;
- same-context, same-number, and matched random-pair comparisons;
- crossed template/value cluster-bootstrap intervals;
- held-out-template and held-out-value ridge probes;
- shuffled-label controls;
- a preprocessing and evidence-boundary audit.

Start with the [expanded full-run report](exp_geometry_probe_validation/20260715_073705_meta-llama_Llama-3.2-1B_full/geometry_probe_validation_expanded_report.md). The full-run `pairwise_measurements.csv` is excluded from Git because of its size; compact CSV summaries and all report figures are tracked.

---

## Additional Control Experiments

- `exp_clinical_specificity/`: clinical-specificity wording variants.
- `exp_hr_wording_flight_baseline/`: `heart rate` versus `HR` wording against a travel baseline.
- `exp_keyword_position/`: domain-keyword placement and token-position confounding.
- `exp_token_position_control/`: approximately position-matched neutral, abbreviated, full-term, and clinical framings.

These controls identify candidate explanations and confounds; they are not confirmatory tests.
