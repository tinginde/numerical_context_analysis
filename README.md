# On the Representation–Behavior Gap of Numerical Information in Large Language Models

**Tzu-Ting (Tina) Chu** | April 2026

A preliminary empirical investigation into why LLMs can generate clinically dangerous numerical values (e.g., body temperature 150 °C) without flagging them as abnormal — and what their internal representations reveal about this failure.

---

## Research Question

Large language models are increasingly deployed in high-stakes domains such as clinical decision support and financial risk assessment. Yet they exhibit systematic failures in numerical reasoning: generating critically abnormal values without apparent awareness of their danger. Are these failures rooted in how numbers are *represented* internally, or only in how they are *decoded* at output?

This project investigates the **Representation–Behavior Gap** in LLM numerical processing: the hypothesis that models encode contextual identity into numerical tokens at the expense of semantic risk distinctions, and that this representational insensitivity propagates into unreliable behavioral outputs.

---

## Key Findings

**Finding 1 — Context-sensitive, value-insensitive.** The same number (e.g., "24") in a medical context ("heart rate 24 bpm") diverges representationally from non-medical uses across layers (final-layer cosine similarity ~0.53). But within a medical context, *different severity levels* (safe vs. critical values) converge to similarity 0.91–1.00 at the final layer. The model distinguishes context type; it does not distinguish numerical risk.

**Finding 2 — The gap is structural, not a capacity limitation.** The convergence pattern holds for both LLaMA 3.2-1B and 3B. Increasing model size improves behavioral performance but does not restore value-discriminative representations. The problem appears to be architectural, not a function of insufficient parameters.

**Finding 3 — Behavioral ordinal awareness is dissociated from representation.** The 3B model achieves Spearman ρ = 0.915 for clinical severity rank ordering in zero-shot mode, yet its representations show near-zero separation between severity levels. The model can rank-order severity without grounding that ranking in its internal encoding — a direct example of the representation–behavior gap.

**Finding 4 — Risk F1 approaches zero under numeric output conditions.** Across all conditions, the model fails to reliably flag clinically critical values. Best overall accuracy is 43.8% (3B, category format, role-prompting), well above 6-class chance (16.7%) but far from deployment-safe.

---

## Experiments

| Script | Experiment | Description |
|--------|-----------|-------------|
| `numerical_context_analysis_v1.py` | Exp A & B | Cross-context and cross-magnitude hidden state analysis. Same number "24" across five semantic contexts (Exp A); same sentence with values 2.4, 24, 240 (Exp B). LLaMA 3.2-1B. |
| `numerical_context_analysis_v2.py` | Exp C (1B) | Medical severity discrimination. Five contexts × four severity levels. Per-layer cosine similarity and final-layer summaries. LLaMA 3.2-1B. |
| `numerical_context_analysis_v3.py` | Exp C (3B) | Same as v2, run on LLaMA 3.2-3B for scale comparison. |
| `numerical_context_analysis_v4.py` | Exp D (PCA) | PCA visualization of hidden states across 8 medical contexts × 6 severity levels. Both 1B and 3B. |
| `numerical_context_analysis_v5.py` | Exp E (Behavior) | Output-level behavioral evaluation. Factorial design: 3 output formats × 4 prompting strategies × 2 models × 8 contexts × 6 severity values = 1,152 inferences. Metrics: accuracy, Spearman ρ, Risk F1, parse rate. |
| `geometry_probe_validation.py` | Geometry/probe validation | Independent validation study for anisotropy-controlled cosine geometry and linear decodability of numeric magnitude. No activation patching, logit lens, or component attribution. |

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Authenticate with HuggingFace (LLaMA 3.2 requires access approval)
huggingface-cli login

# 3. Run experiments
python numerical_context_analysis_v1.py   # Cross-context + cross-magnitude (Exp A & B)
python numerical_context_analysis_v2.py   # Medical severity, 1B (Exp C)
python numerical_context_analysis_v3.py   # Medical severity, 3B (Exp C)
python numerical_context_analysis_v4.py   # PCA visualization (Exp D)
python numerical_context_analysis_v5.py   # Behavioral evaluation, 1,152 inferences (Exp E)
python geometry_probe_validation.py       # Geometry/probe validation, full stimulus set
```

All experiments use greedy decoding and the model's native tokenizer to control for stochasticity and tokenization artifacts. Output figures are saved to `results/`.

### Geometry/probe validation

The geometry/probe validation experiment uses plain-text prompts (no chat template). It asks whether apparent late-layer convergence among different
numbers in the same context remains after simple anisotropy controls, and
whether numeric magnitude remains linearly decodable from residual-stream
activations.

The full-run default is the gated base checkpoint `meta-llama/Llama-3.2-1B`,
not the Instruct checkpoint. Before the first run, accept access for that model
on Hugging Face, then authenticate locally:

```bash
huggingface-cli login
python geometry_probe_validation.py --validate-load-only
```

The validation command loads only the checkpoint and prints its resolved
revision, parameter count, dtype, device, Transformers version, seed, and run
mode. It never falls back to a tiny/random model: authentication, access, or
download failures stop with an explicit error.

After the load-only validation succeeds, run the complete 225-prompt experiment:

```bash
python geometry_probe_validation.py --model meta-llama/Llama-3.2-1B --batch-size 4
```

CUDA is selected automatically when available and uses bfloat16 when supported;
extracted activations are converted to float32 before cosine correction and
probe fitting. The config and report record the resolved checkpoint metadata.

For a small CPU-only pipeline check, explicitly opt into the tiny random model:

```bash
python geometry_probe_validation.py --smoke-test --device cpu
```

Outputs are written under a timestamped directory in
`results/exp_geometry_probe_validation/` and include:

- `stimuli.csv`: balanced generated prompts with validated number-token spans.
- `pairwise_measurements.csv`: sampled raw, centered, and standardized cosine measurements.
- `pairwise_summary.csv`: compact layer-wise summaries by pair category.
- `pairwise_regression.csv`: descriptive regressions of similarity on same-context, same-number, and absolute numeric distance.
- `probe_results.csv`: ridge probe R2 and MAE for held-out-template and held-out-value splits, including shuffled-label controls.
- `geometry_probe_validation_report.md`: concise interpretation with explicit limitations.

Limitations: this is a geometry-and-probing validation only. It does not test
causal use of numeric magnitude, task-dependent construction, attention-head
attribution, MLP attribution, normalization ablations, or activation patching.
---

## Repository Structure

```
numerical_context_analysis/
├── numerical_context_analysis_v1.py   # Exp A & B: cross-context, cross-magnitude
├── numerical_context_analysis_v2.py   # Exp C: medical severity, LLaMA 1B
├── numerical_context_analysis_v3.py   # Exp C: medical severity, LLaMA 3B
├── numerical_context_analysis_v4.py   # Exp D: PCA visualization
├── numerical_context_analysis_v5.py   # Exp E: behavioral evaluation
├── v5_generate_analysis.py            # Post-hoc analysis for Exp E outputs
├── requirements.txt
├── results/
│   ├── exp_A_B/                       # Exp A & B figures
│   ├── exp_C/                         # Exp C figures (1B and 3B)
│   ├── exp_D/                         # Exp D figures
│   └── exp_E/                         # Exp E figures, metrics CSV, raw CSV
└── README.md
```

---

## Relation to Research Proposal

This repository accompanies the research proposal *"On the Representation–Behavior Gap of Numerical Information in Large Language Models"* and the preliminary experimental report (April 2026). The experiments here establish three foundational observations:

1. Numbers in LLMs are context-sensitive but value-insensitive in deep layers.
2. This insensitivity propagates to behavioral outputs in clinically relevant settings.
3. The gap is present across model scales, motivating causal analysis (activation patching) as the next step.

The formal research phase will extend this work to LLaMA 3.2-7B, introduce linear probing and activation patching, and develop a systematic perturbation benchmark across medical, financial, and scientific numerical domains.

---

## Citation

If you use this code or findings in your work, please cite:

```
Chu, T.-T. (2026). On the Representation–Behavior Gap of Numerical Information
in Large Language Models: Preliminary Experimental Report. Unpublished manuscript.
GitHub: https://github.com/tinginde/numerical_context_analysis
```

