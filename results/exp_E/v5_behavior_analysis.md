# v5 Behavior Analysis

## Scope
This report analyzes the results/exp_E outputs.

## Executive Summary
- 1B numeric behavior is dominated by refusal first and central-value collapse second. The highest refusal condition is 1B numeric zero_shot at 0.85 refusal rate.
- The strongest overall setting is 3B category role_prompting with 0.458 accuracy.
- The strongest ranking signal in scalar scoring is 1B numeric role_prompting with Spearman rho 0.889, showing ordinal awareness without strong calibration.
- Category outputs are more useful than numeric outputs for this task because the model can align better with biomarker-specific labels than with exact scalar danger levels.

## Behavioral Findings
### 1B
- Zero-shot and role-prompted numeric runs are bottlenecked by safety-style refusals rather than pure reasoning failure.
- Once forced to answer, 1B usually predicts the middle score 2, which collapses extreme low and extreme high cases back toward Normal.
- Category outputs parse cleanly, but accuracy stays close to chance, so instruction following does not imply clinical mapping competence.

### 3B
- 3B almost never refuses and shows stable output formatting across all three response formats.
- Numeric outputs have moderate rank-order sensitivity but systematic upward bias, often preferring scores 3 or 4.
- Category outputs are the cleanest expression of the model's knowledge. They outperform numeric and hybrid modes in both accuracy and usability.

## Context Pattern
Best 3B category contexts under role prompting:
- Creatinine: accuracy 0.667
- Glucose: accuracy 0.667
- Temp: accuracy 0.667

Weakest 3B category contexts under role prompting:
- Insulin: accuracy 0.000
- HbA1c: accuracy 0.333
- TG: accuracy 0.333

## Interpretation
- Small instruction-tuned models are dominated by policy refusal and then by a regression-to-normal fallback once refusal is suppressed.
- Larger models preserve the ordinal shape of clinical risk better than they preserve exact level calibration.
- Hybrid outputs improve self-consistency in 3B, but self-consistency does not guarantee correctness. The model can be confidently wrong in both the digit and label at the same time.

## Files Generated
- v5_behavior_summary.csv
- v5_context_breakdown.csv
- v5_error_cases.csv
- v5_behavior_overview.png