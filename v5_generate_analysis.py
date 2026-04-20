from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULTS_DIR = Path(__file__).resolve().parent / "results" / "exp_E"
RAW_PATH = RESULTS_DIR / "v5_raw.csv"
METRICS_PATH = RESULTS_DIR / "v5_metrics.csv"

REPORT_PATH = RESULTS_DIR / "v5_behavior_analysis.md"
SUMMARY_PATH = RESULTS_DIR / "v5_behavior_summary.csv"
CONTEXT_PATH = RESULTS_DIR / "v5_context_breakdown.csv"
ERROR_CASES_PATH = RESULTS_DIR / "v5_error_cases.csv"
FIGURE_PATH = RESULTS_DIR / "v5_behavior_overview.png"


def load_data():
    raw = pd.read_csv(RAW_PATH)
    metrics = pd.read_csv(METRICS_PATH)
    return raw, metrics


def build_summary_tables(raw: pd.DataFrame, metrics: pd.DataFrame):
    summary = metrics.copy()

    numeric = raw[raw["format"] == "numeric"].copy()
    numeric["correct"] = numeric["numeric"] == numeric["ground_truth"]
    numeric["signed_error"] = numeric["numeric"] - numeric["ground_truth"]
    numeric["absolute_error"] = (numeric["numeric"] - numeric["ground_truth"]).abs()
    numeric["refusal"] = numeric["raw_output"].fillna("").str.contains(
        r"cannot|can't provide|consult|medical professional|anything else",
        case=False,
        regex=True,
    )

    numeric_stats = (
        numeric.groupby(["model", "format", "role"])
        .apply(
            lambda g: pd.Series(
                {
                    "accuracy_all_rows": g["correct"].fillna(False).mean(),
                    "accuracy_parsed_only": g.loc[g["numeric"].notna(), "correct"].mean()
                    if g["numeric"].notna().any()
                    else np.nan,
                    "refusal_rate": g["refusal"].mean(),
                    "mean_prediction": g["numeric"].dropna().mean() if g["numeric"].notna().any() else np.nan,
                    "mode_prediction": g["numeric"].dropna().mode().iloc[0] if g["numeric"].notna().any() else np.nan,
                    "mean_signed_error": g["signed_error"].dropna().mean() if g["signed_error"].notna().any() else np.nan,
                    "mean_absolute_error": g["absolute_error"].dropna().mean() if g["absolute_error"].notna().any() else np.nan,
                }
            )
        )
        .reset_index()
    )

    hybrid = raw[raw["format"] == "hybrid"].copy()
    hybrid["hybrid_numeric_correct"] = hybrid["numeric"] == hybrid["ground_truth"]
    hybrid["hybrid_category_correct"] = hybrid["category"] == hybrid["ground_truth"]
    hybrid_stats = (
        hybrid.groupby(["model", "format", "role"])
        .apply(
            lambda g: pd.Series(
                {
                    "hybrid_num_acc": g["hybrid_numeric_correct"].fillna(False).mean(),
                    "hybrid_cat_acc": g["hybrid_category_correct"].fillna(False).mean(),
                    "hybrid_consistency_observed": g["hybrid_consistent"].dropna().mean()
                    if g["hybrid_consistent"].notna().any()
                    else np.nan,
                }
            )
        )
        .reset_index()
    )

    summary = summary.merge(numeric_stats, on=["model", "format", "role"], how="left")
    summary = summary.merge(hybrid_stats, on=["model", "format", "role"], how="left")
    summary = summary.sort_values(["model", "format", "role"]).reset_index(drop=True)

    context_breakdown = []
    for fmt in ["numeric", "category", "hybrid"]:
        subset = raw[raw["format"] == fmt].copy()
        if fmt == "numeric":
            subset["parsed"] = subset["numeric"].notna()
            subset["correct"] = subset["numeric"] == subset["ground_truth"]
            grouped = (
                subset.groupby(["model", "role", "context"])
                .apply(
                    lambda g: pd.Series(
                        {
                            "format": fmt,
                            "parse_rate": g["parsed"].mean(),
                            "accuracy": g["correct"].fillna(False).mean(),
                            "mean_prediction": g["numeric"].dropna().mean() if g["numeric"].notna().any() else np.nan,
                            "mode_prediction": g["numeric"].dropna().mode().iloc[0] if g["numeric"].notna().any() else np.nan,
                        }
                    )
                )
                .reset_index()
            )
        elif fmt == "category":
            subset["correct"] = subset["category"] == subset["ground_truth"]
            grouped = (
                subset.groupby(["model", "role", "context"])
                .apply(
                    lambda g: pd.Series(
                        {
                            "format": fmt,
                            "parse_rate": g["category"].notna().mean(),
                            "accuracy": g["correct"].fillna(False).mean(),
                            "mean_prediction": np.nan,
                            "mode_prediction": np.nan,
                        }
                    )
                )
                .reset_index()
            )
        else:
            subset["correct_numeric"] = subset["numeric"] == subset["ground_truth"]
            subset["correct_category"] = subset["category"] == subset["ground_truth"]
            grouped = (
                subset.groupby(["model", "role", "context"])
                .apply(
                    lambda g: pd.Series(
                        {
                            "format": fmt,
                            "parse_rate": g["numeric"].notna().mean(),
                            "accuracy": g["correct_numeric"].fillna(False).mean(),
                            "category_accuracy": g["correct_category"].fillna(False).mean(),
                            "hybrid_consistency": g["hybrid_consistent"].dropna().mean()
                            if g["hybrid_consistent"].notna().any()
                            else np.nan,
                            "mean_prediction": g["numeric"].dropna().mean() if g["numeric"].notna().any() else np.nan,
                            "mode_prediction": g["numeric"].dropna().mode().iloc[0] if g["numeric"].notna().any() else np.nan,
                        }
                    )
                )
                .reset_index()
            )
        context_breakdown.append(grouped)

    context_df = pd.concat(context_breakdown, ignore_index=True)
    return summary, context_df, numeric


def select_error_cases(raw: pd.DataFrame):
    cases = []

    numeric = raw[raw["format"] == "numeric"].copy()
    numeric["refusal"] = numeric["raw_output"].fillna("").str.contains(
        r"cannot|can't provide|consult|medical professional|anything else",
        case=False,
        regex=True,
    )

    refusal_cases = numeric[
        (numeric["model"] == "1B")
        & (numeric["role"].isin(["zero_shot", "role_prompting"]))
        & (numeric["refusal"])
    ].head(6)
    for _, row in refusal_cases.iterrows():
        cases.append(
            {
                "behavior_type": "refusal",
                "model": row["model"],
                "format": row["format"],
                "role": row["role"],
                "context": row["context"],
                "value": row["value"],
                "ground_truth": row["ground_truth"],
                "gt_label": row["gt_label"],
                "pred_numeric": row["numeric"],
                "pred_category": row["category"],
                "note": "Model refused to provide a clinical score.",
                "raw_output": row["raw_output"],
            }
        )

    collapse_cases = numeric[
        (numeric["model"] == "1B")
        & (numeric["role"].isin(["forced_answer", "instruction_constrained"]))
        & (numeric["numeric"] == 2)
        & (numeric["ground_truth"].isin([0, 5]))
    ].head(6)
    for _, row in collapse_cases.iterrows():
        cases.append(
            {
                "behavior_type": "central_collapse",
                "model": row["model"],
                "format": row["format"],
                "role": row["role"],
                "context": row["context"],
                "value": row["value"],
                "ground_truth": row["ground_truth"],
                "gt_label": row["gt_label"],
                "pred_numeric": row["numeric"],
                "pred_category": row["category"],
                "note": "Extreme value was mapped back to the neutral middle score 2.",
                "raw_output": row["raw_output"],
            }
        )

    overestimate_pool = numeric[
        (numeric["model"] == "3B")
        & (numeric["numeric"].notna())
        & ((numeric["numeric"] - numeric["ground_truth"]) >= 2)
    ].copy()
    overestimate_pool["error_size"] = overestimate_pool["numeric"] - overestimate_pool["ground_truth"]
    overestimate_cases = overestimate_pool.sort_values(["error_size", "role", "context"], ascending=[False, True, True]).head(6)
    for _, row in overestimate_cases.iterrows():
        cases.append(
            {
                "behavior_type": "risk_overestimation",
                "model": row["model"],
                "format": row["format"],
                "role": row["role"],
                "context": row["context"],
                "value": row["value"],
                "ground_truth": row["ground_truth"],
                "gt_label": row["gt_label"],
                "pred_numeric": row["numeric"],
                "pred_category": row["category"],
                "note": "Model produced a risk score at least two levels above ground truth.",
                "raw_output": row["raw_output"],
            }
        )

    category = raw[raw["format"] == "category"].copy()
    category_errors = category[
        (category["model"] == "3B")
        & (category["role"].isin(["zero_shot", "role_prompting"]))
        & (category["category"].notna())
        & (category["category"] != category["ground_truth"])
    ].head(6)
    for _, row in category_errors.iterrows():
        cases.append(
            {
                "behavior_type": "label_mismatch",
                "model": row["model"],
                "format": row["format"],
                "role": row["role"],
                "context": row["context"],
                "value": row["value"],
                "ground_truth": row["ground_truth"],
                "gt_label": row["gt_label"],
                "pred_numeric": row["numeric"],
                "pred_category": row["category"],
                "note": "Model returned a valid biomarker label, but the wrong one.",
                "raw_output": row["raw_output"],
            }
        )

    return pd.DataFrame(cases)


def build_report(summary: pd.DataFrame, context_df: pd.DataFrame):
    numeric_rows = summary[summary["format"] == "numeric"].copy()
    category_rows = summary[summary["format"] == "category"].copy()
    hybrid_rows = summary[summary["format"] == "hybrid"].copy()

    best_category = category_rows.sort_values("accuracy", ascending=False).iloc[0]
    best_numeric = numeric_rows.sort_values("spearman_rho", ascending=False).iloc[0]
    worst_refusal = numeric_rows.sort_values("refusal_rate", ascending=False).iloc[0]

    category_context = context_df[context_df["format"] == "category"].copy()
    top_contexts = (
        category_context[(category_context["model"] == "3B") & (category_context["role"] == "role_prompting")]
        .sort_values("accuracy", ascending=False)
        .head(3)
    )
    weak_contexts = (
        category_context[(category_context["model"] == "3B") & (category_context["role"] == "role_prompting")]
        .sort_values("accuracy", ascending=True)
        .head(3)
    )

    lines = [
        "# v5 Behavior Analysis",
        "",
        "## Scope",
        "This report analyzes the results/exp_E outputs.",
        "",
        "## Executive Summary",
        f"- 1B numeric behavior is dominated by refusal first and central-value collapse second. The highest refusal condition is {worst_refusal['model']} {worst_refusal['format']} {worst_refusal['role']} at {worst_refusal['refusal_rate']:.2f} refusal rate.",
        f"- The strongest overall setting is {best_category['model']} {best_category['format']} {best_category['role']} with {best_category['accuracy']:.3f} accuracy.",
        f"- The strongest ranking signal in scalar scoring is {best_numeric['model']} {best_numeric['format']} {best_numeric['role']} with Spearman rho {best_numeric['spearman_rho']:.3f}, showing ordinal awareness without strong calibration.",
        "- Category outputs are more useful than numeric outputs for this task because the model can align better with biomarker-specific labels than with exact scalar danger levels.",
        "",
        "## Behavioral Findings",
        "### 1B",
        "- Zero-shot and role-prompted numeric runs are bottlenecked by safety-style refusals rather than pure reasoning failure.",
        "- Once forced to answer, 1B usually predicts the middle score 2, which collapses extreme low and extreme high cases back toward Normal.",
        "- Category outputs parse cleanly, but accuracy stays close to chance, so instruction following does not imply clinical mapping competence.",
        "",
        "### 3B",
        "- 3B almost never refuses and shows stable output formatting across all three response formats.",
        "- Numeric outputs have moderate rank-order sensitivity but systematic upward bias, often preferring scores 3 or 4.",
        "- Category outputs are the cleanest expression of the model's knowledge. They outperform numeric and hybrid modes in both accuracy and usability.",
        "",
        "## Context Pattern",
        "Best 3B category contexts under role prompting:",
    ]

    for _, row in top_contexts.iterrows():
        lines.append(f"- {row['context']}: accuracy {row['accuracy']:.3f}")

    lines.extend([
        "",
        "Weakest 3B category contexts under role prompting:",
    ])

    for _, row in weak_contexts.iterrows():
        lines.append(f"- {row['context']}: accuracy {row['accuracy']:.3f}")

    lines.extend([
        "",
        "## Interpretation",
        "- Small instruction-tuned models are dominated by policy refusal and then by a regression-to-normal fallback once refusal is suppressed.",
        "- Larger models preserve the ordinal shape of clinical risk better than they preserve exact level calibration.",
        "- Hybrid outputs improve self-consistency in 3B, but self-consistency does not guarantee correctness. The model can be confidently wrong in both the digit and label at the same time.",
        "",
        "## Files Generated",
        "- v5_behavior_summary.csv",
        "- v5_context_breakdown.csv",
        "- v5_error_cases.csv",
        "- v5_behavior_overview.png",
    ])

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def build_figure(summary: pd.DataFrame):
    numeric = summary[summary["format"] == "numeric"].copy()
    numeric["label"] = numeric["model"] + "-" + numeric["role"]

    role_order = ["zero_shot", "role_prompting", "instruction_constrained", "forced_answer"]
    model_order = ["1B", "3B"]
    labels = [f"{model}\n{role.replace('_', ' ')}" for model in model_order for role in role_order]

    ordered = []
    for model in model_order:
        for role in role_order:
            ordered.append(numeric[(numeric["model"] == model) & (numeric["role"] == role)].iloc[0])
    ordered = pd.DataFrame(ordered)

    x = np.arange(len(ordered))
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)

    axes[0].bar(x, ordered["refusal_rate"], color=["#4C78A8" if m == "1B" else "#E45756" for m in ordered["model"]])
    axes[0].set_title("Refusal Rate")
    axes[0].set_ylabel("rate")
    axes[0].set_xticks(x, labels, rotation=0)
    axes[0].set_ylim(0, max(0.4, float(ordered["refusal_rate"].max()) + 0.05))

    axes[1].bar(x, ordered["mean_prediction"], color=["#4C78A8" if m == "1B" else "#E45756" for m in ordered["model"]])
    axes[1].axhline(2.5, color="#666666", linestyle="--", linewidth=1)
    axes[1].set_title("Mean Numeric Prediction")
    axes[1].set_ylabel("predicted level")
    axes[1].set_xticks(x, labels, rotation=0)
    axes[1].set_ylim(0, 5)

    axes[2].bar(x, ordered["mean_signed_error"], color=["#4C78A8" if m == "1B" else "#E45756" for m in ordered["model"]])
    axes[2].axhline(0, color="#666666", linestyle="--", linewidth=1)
    axes[2].set_title("Signed Error")
    axes[2].set_ylabel("prediction - truth")
    axes[2].set_xticks(x, labels, rotation=0)

    fig.suptitle("v5 Behavior Overview: refusal, collapse, and risk bias")
    fig.savefig(FIGURE_PATH, dpi=220)
    plt.close(fig)


def main():
    raw, metrics = load_data()
    summary, context_df, _numeric = build_summary_tables(raw, metrics)
    error_cases = select_error_cases(raw)

    summary.round(4).to_csv(SUMMARY_PATH, index=False)
    context_df.round(4).to_csv(CONTEXT_PATH, index=False)
    error_cases.to_csv(ERROR_CASES_PATH, index=False)
    build_report(summary, context_df)
    build_figure(summary)

    print(f"Wrote {REPORT_PATH.name}")
    print(f"Wrote {SUMMARY_PATH.name}")
    print(f"Wrote {CONTEXT_PATH.name}")
    print(f"Wrote {ERROR_CASES_PATH.name}")
    print(f"Wrote {FIGURE_PATH.name}")


if __name__ == "__main__":
    main()