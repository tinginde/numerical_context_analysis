"""Regenerate the geometry/probe report from completed CSV outputs only.

This script deliberately has no model-loading or activation-extraction path.  It
adds crossed template/value cluster-bootstrap uncertainty estimates, regression
coefficient intervals, split documentation, and expanded per-layer tables.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PAIR_CATEGORIES = [
    "same_context_different_number",
    "same_number_different_context",
    "same_context_same_number_different_template",
    "random_baseline",
]
NON_BASELINE_CATEGORIES = PAIR_CATEGORIES[:-1]
CATEGORY_LABELS = {
    "same_context_different_number": "SC-DN",
    "same_number_different_context": "SN-DC",
    "same_context_same_number_different_template": "SC-SN-DT",
    "random_baseline": "Random",
}
POSITIONS = ["numeral_final", "prompt_final"]
TRANSFORMS = ["raw", "centered", "standardized"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Completed run directory containing CSV outputs")
    parser.add_argument("--bootstrap-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def percentile_interval(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return float("nan"), float("nan")
    return tuple(np.percentile(finite, [2.5, 97.5]).tolist())


def fmt(x: float, digits: int = 3) -> str:
    if not np.isfinite(float(x)):
        return "NA"
    return f"{float(x):.{digits}f}"


def md_table(headers: list[str], rows: Iterable[Iterable[object]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    return lines


def read_inputs(run_dir: Path) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required = [
        "config.json",
        "stimuli.csv",
        "pairwise_measurements.csv",
        "pairwise_regression.csv",
        "probe_results.csv",
    ]
    missing = [name for name in required if not (run_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required completed outputs: {missing}")
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    stimuli = pd.read_csv(run_dir / "stimuli.csv")
    pairs = pd.read_csv(run_dir / "pairwise_measurements.csv")
    old_regression = pd.read_csv(run_dir / "pairwise_regression.csv")
    probes = pd.read_csv(run_dir / "probe_results.csv")
    return config, stimuli, pairs, old_regression, probes


def add_cluster_ids(pairs: pd.DataFrame, stimuli: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    meta = stimuli[["stimulus_id", "context", "template_id", "value"]].copy()
    meta["template_unit"] = meta["context"].astype(str) + "|" + meta["template_id"].astype(str)
    template_levels = sorted(meta["template_unit"].unique())
    value_levels = sorted(meta["value"].astype(int).unique())
    template_map = {value: i for i, value in enumerate(template_levels)}
    value_map = {value: i for i, value in enumerate(value_levels)}
    meta["template_cluster"] = meta["template_unit"].map(template_map).astype(np.int16)
    meta["value_cluster"] = meta["value"].astype(int).map(value_map).astype(np.int16)

    endpoint = meta[["stimulus_id", "template_cluster", "value_cluster"]]
    pairs = pairs.merge(endpoint.add_suffix("_a"), on="stimulus_id_a", how="left", validate="many_to_one")
    pairs = pairs.merge(endpoint.add_suffix("_b"), on="stimulus_id_b", how="left", validate="many_to_one")
    if pairs[["template_cluster_a", "value_cluster_a", "template_cluster_b", "value_cluster_b"]].isna().any().any():
        raise ValueError("Could not map every pair endpoint back to stimuli.csv")
    return pairs, len(template_levels), len(value_levels)


def make_crossed_cluster_weights(
    base: pd.DataFrame,
    n_templates: int,
    n_values: int,
    samples: int,
    seed: int,
) -> np.ndarray:
    """Two-way crossed bootstrap over context-specific templates and values.

    Each replicate resamples the 15 context-specific templates and 15 numeric
    values independently.  A pair receives the product of the resampling
    multiplicities of both endpoints.  This preserves dyadic dependence rather
    than pretending 25,200 pair rows are independent observations.
    """
    rng = np.random.default_rng(seed)
    template_counts = np.empty((samples, n_templates), dtype=np.float32)
    value_counts = np.empty((samples, n_values), dtype=np.float32)
    for b in range(samples):
        template_counts[b] = np.bincount(rng.integers(0, n_templates, n_templates), minlength=n_templates)
        value_counts[b] = np.bincount(rng.integers(0, n_values, n_values), minlength=n_values)
    ta = base["template_cluster_a"].to_numpy(dtype=int)
    tb = base["template_cluster_b"].to_numpy(dtype=int)
    va = base["value_cluster_a"].to_numpy(dtype=int)
    vb = base["value_cluster_b"].to_numpy(dtype=int)
    return template_counts[:, ta] * template_counts[:, tb] * value_counts[:, va] * value_counts[:, vb]


def verify_repeated_pair_order(groups: list[tuple[tuple, pd.DataFrame]], base: pd.DataFrame) -> None:
    base_a = base["stimulus_id_a"].to_numpy()
    base_b = base["stimulus_id_b"].to_numpy()
    for key, group in groups[1:]:
        if len(group) != len(base) or not np.array_equal(group["stimulus_id_a"].to_numpy(), base_a) or not np.array_equal(group["stimulus_id_b"].to_numpy(), base_b):
            raise ValueError(f"Pair ordering differs for group {key}; refusing to reuse bootstrap weights")


def cluster_bootstrap_analysis(
    pairs: pd.DataFrame,
    samples: int,
    seed: int,
    n_templates: int,
    n_values: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    group_cols = ["position", "layer", "transform"]
    groups = list(pairs.groupby(group_cols, sort=False, observed=True))
    if not groups:
        raise ValueError("pairwise_measurements.csv contains no groups")
    base = groups[0][1].reset_index(drop=True)
    verify_repeated_pair_order(groups, base)

    weights = make_crossed_cluster_weights(base, n_templates, n_values, samples, seed).astype(np.float64)
    categories = base["pair_category"].to_numpy()
    category_masks = {cat: (categories == cat).astype(np.float64) for cat in PAIR_CATEGORIES}
    category_denominators = {cat: weights @ mask for cat, mask in category_masks.items()}

    x = np.column_stack(
        [
            np.ones(len(base)),
            base["same_context"].to_numpy(dtype=float),
            base["same_number"].to_numpy(dtype=float),
            base["abs_delta_value"].to_numpy(dtype=float),
        ]
    )
    xtwx = np.einsum("bn,ni,nj->bij", weights, x, x, optimize=True)
    xtwx_inv = np.asarray([np.linalg.pinv(matrix) for matrix in xtwx])
    weight_sums = weights.sum(axis=1)

    summary_rows: list[dict] = []
    relative_rows: list[dict] = []
    regression_rows: list[dict] = []

    for (position, layer, transform), group in groups:
        y = group["similarity"].to_numpy(dtype=float)
        bootstrap_category_means: dict[str, np.ndarray] = {}
        for cat in PAIR_CATEGORIES:
            mask = category_masks[cat]
            numerators = weights @ (y * mask)
            means = np.divide(
                numerators,
                category_denominators[cat],
                out=np.full(samples, np.nan),
                where=category_denominators[cat] > 0,
            )
            bootstrap_category_means[cat] = means
            original = y[categories == cat]
            lo, hi = percentile_interval(means)
            summary_rows.append(
                {
                    "position": position,
                    "layer": int(layer),
                    "transform": transform,
                    "pair_category": cat,
                    "mean_similarity": float(original.mean()),
                    "cluster_bootstrap_ci_low": lo,
                    "cluster_bootstrap_ci_high": hi,
                    "n_pair_rows": int(len(original)),
                    "n_template_clusters": n_templates,
                    "n_value_clusters": n_values,
                    "bootstrap_samples": samples,
                }
            )

        baseline_original = y[categories == "random_baseline"].mean()
        baseline_bootstrap = bootstrap_category_means["random_baseline"]
        for cat in NON_BASELINE_CATEGORIES:
            delta_bootstrap = bootstrap_category_means[cat] - baseline_bootstrap
            lo, hi = percentile_interval(delta_bootstrap)
            relative_rows.append(
                {
                    "position": position,
                    "layer": int(layer),
                    "transform": transform,
                    "pair_category": cat,
                    "mean_difference_from_random": float(y[categories == cat].mean() - baseline_original),
                    "cluster_bootstrap_ci_low": lo,
                    "cluster_bootstrap_ci_high": hi,
                    "n_category_pair_rows": int((categories == cat).sum()),
                    "n_random_pair_rows": int((categories == "random_baseline").sum()),
                    "n_template_clusters": n_templates,
                    "n_value_clusters": n_values,
                    "bootstrap_samples": samples,
                }
            )

        original_beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        original_pred = x @ original_beta
        original_r2 = 1.0 - np.square(y - original_pred).sum() / np.square(y - y.mean()).sum()
        xtwy = np.einsum("bn,ni,n->bi", weights, x, y, optimize=True)
        beta_boot = np.einsum("bij,bj->bi", xtwx_inv, xtwy, optimize=True)
        sum_y = weights @ y
        sum_y2 = weights @ np.square(y)
        sst = sum_y2 - np.square(sum_y) / weight_sums
        sse = sum_y2 - 2 * np.einsum("bi,bi->b", beta_boot, xtwy) + np.einsum(
            "bi,bij,bj->b", beta_boot, xtwx, beta_boot, optimize=True
        )
        r2_boot = np.divide(1.0 * (sst - sse), sst, out=np.full(samples, np.nan), where=sst > 0)
        intervals = [percentile_interval(beta_boot[:, j]) for j in range(4)]
        r2_lo, r2_hi = percentile_interval(r2_boot)
        regression_rows.append(
            {
                "position": position,
                "layer": int(layer),
                "transform": transform,
                "intercept": float(original_beta[0]),
                "intercept_ci_low": intervals[0][0],
                "intercept_ci_high": intervals[0][1],
                "coef_same_context": float(original_beta[1]),
                "coef_same_context_ci_low": intervals[1][0],
                "coef_same_context_ci_high": intervals[1][1],
                "coef_same_number": float(original_beta[2]),
                "coef_same_number_ci_low": intervals[2][0],
                "coef_same_number_ci_high": intervals[2][1],
                "coef_abs_delta_value": float(original_beta[3]),
                "coef_abs_delta_value_ci_low": intervals[3][0],
                "coef_abs_delta_value_ci_high": intervals[3][1],
                "r2_descriptive": float(original_r2),
                "r2_ci_low": r2_lo,
                "r2_ci_high": r2_hi,
                "n_pair_rows": int(len(y)),
                "n_template_clusters": n_templates,
                "n_value_clusters": n_values,
                "bootstrap_samples": samples,
            }
        )

    return pd.DataFrame(summary_rows), pd.DataFrame(relative_rows), pd.DataFrame(regression_rows)


def geometry_plots(summary: pd.DataFrame, relative: pd.DataFrame, run_dir: Path) -> None:
    colors = {
        "same_context_different_number": "#1f77b4",
        "same_number_different_context": "#ff7f0e",
        "same_context_same_number_different_template": "#2ca02c",
        "random_baseline": "#6b7280",
    }
    for position in POSITIONS:
        fig, axes = plt.subplots(3, 2, figsize=(14, 13), sharex=True)
        for row_index, transform in enumerate(TRANSFORMS):
            ax_mean, ax_delta = axes[row_index]
            for cat in PAIR_CATEGORIES:
                rows = summary[(summary.position == position) & (summary.transform == transform) & (summary.pair_category == cat)].sort_values("layer")
                x = rows.layer.to_numpy()
                y = rows.mean_similarity.to_numpy()
                lo = rows.cluster_bootstrap_ci_low.to_numpy()
                hi = rows.cluster_bootstrap_ci_high.to_numpy()
                ax_mean.plot(x, y, marker="o", ms=3, label=CATEGORY_LABELS[cat], color=colors[cat])
                ax_mean.fill_between(x, lo, hi, color=colors[cat], alpha=0.12)
            for cat in NON_BASELINE_CATEGORIES:
                rows = relative[(relative.position == position) & (relative.transform == transform) & (relative.pair_category == cat)].sort_values("layer")
                x = rows.layer.to_numpy()
                y = rows.mean_difference_from_random.to_numpy()
                lo = rows.cluster_bootstrap_ci_low.to_numpy()
                hi = rows.cluster_bootstrap_ci_high.to_numpy()
                ax_delta.plot(x, y, marker="o", ms=3, label=CATEGORY_LABELS[cat], color=colors[cat])
                ax_delta.fill_between(x, lo, hi, color=colors[cat], alpha=0.12)
            ax_delta.axhline(0, color="black", linewidth=1, alpha=0.5)
            ax_mean.set_ylabel(f"{transform}\nmean cosine")
            ax_delta.set_ylabel(f"{transform}\ndifference from random")
            ax_mean.grid(alpha=0.2)
            ax_delta.grid(alpha=0.2)
        axes[0, 0].set_title("Pair-category cosine")
        axes[0, 1].set_title("Category minus random baseline")
        axes[-1, 0].set_xlabel("Layer")
        axes[-1, 1].set_xlabel("Layer")
        axes[0, 0].legend(fontsize=8)
        axes[0, 1].legend(fontsize=8)
        fig.suptitle(f"{position}: crossed template/value cluster bootstrap (95% CI)")
        fig.tight_layout()
        fig.savefig(run_dir / f"cluster_geometry_{position}.png", dpi=180)
        plt.close(fig)


def template_fold_details(stimuli: pd.DataFrame) -> list[dict]:
    details = []
    for template_index in sorted(stimuli.template_index.unique()):
        test = stimuli[stimuli.template_index == template_index]
        train = stimuli[stimuli.template_index != template_index]
        details.append(
            {
                "fold": f"T{int(template_index) + 1}",
                "test_templates": [
                    prompt.replace(str(int(value)), "{value}")
                    for prompt, value in test.groupby("context", sort=True)[["prompt", "value"]].first().itertuples(index=False)
                ],
                "train_templates": sorted(train.template_id.unique()),
                "train_values": sorted(train.value.astype(int).unique()),
                "test_values": sorted(test.value.astype(int).unique()),
                "n_train": len(train),
                "n_test": len(test),
            }
        )
    return details


def value_fold_details(stimuli: pd.DataFrame) -> list[dict]:
    values = sorted(stimuli.value.astype(int).unique())
    inner = values[1:-1]
    test_folds = [inner[::2], inner[1::2] or inner[:1]]
    details = []
    for index, test_values in enumerate(test_folds, 1):
        test = stimuli[stimuli.value.astype(int).isin(test_values)]
        train = stimuli[~stimuli.value.astype(int).isin(test_values)]
        details.append(
            {
                "fold": index,
                "train_values": sorted(train.value.astype(int).unique()),
                "test_values": test_values,
                "train_templates": sorted((train.context.astype(str) + "/" + train.template_id.astype(str)).unique()),
                "test_templates": sorted((test.context.astype(str) + "/" + test.template_id.astype(str)).unique()),
                "n_train": len(train),
                "n_test": len(test),
            }
        )
    return details


def selected_layers(all_layers: list[int]) -> tuple[list[int], int, list[int]]:
    middle = all_layers[len(all_layers) // 2]
    quartile_count = int(np.ceil(len(all_layers) / 4))
    final_quartile = all_layers[-quartile_count:]
    selected = sorted(set([all_layers[0], middle, *final_quartile, all_layers[-1]]))
    return selected, middle, final_quartile


def pair_cell(row: pd.Series) -> str:
    return f"{fmt(row.mean_similarity)} [{fmt(row.cluster_bootstrap_ci_low)}, {fmt(row.cluster_bootstrap_ci_high)}]; n={int(row.n_pair_rows)}"


def relative_cell(row: pd.Series) -> str:
    return f"{fmt(row.mean_difference_from_random)} [{fmt(row.cluster_bootstrap_ci_low)}, {fmt(row.cluster_bootstrap_ci_high)}]"


def coef_cell(row: pd.Series, name: str) -> str:
    return f"{fmt(row[name])} [{fmt(row[name + '_ci_low'])}, {fmt(row[name + '_ci_high'])}]"


def append_geometry_tables(lines: list[str], summary: pd.DataFrame, layers: list[int], heading: str) -> None:
    lines.extend([f"### {heading}", "", "Cells are mean cosine [cluster-bootstrap 95% CI]; n.", ""])
    for position in POSITIONS:
        for transform in TRANSFORMS:
            subset = summary[(summary.position == position) & (summary.transform == transform) & (summary.layer.isin(layers))]
            pivot = {int(layer): {row.pair_category: row for _, row in group.iterrows()} for layer, group in subset.groupby("layer")}
            rows = []
            for layer in layers:
                by_cat = pivot[layer]
                rows.append([layer] + [pair_cell(by_cat[cat]) for cat in PAIR_CATEGORIES])
            lines.extend([f"**{position} — {transform}**", ""])
            lines.extend(md_table(["Layer"] + [CATEGORY_LABELS[c] for c in PAIR_CATEGORIES], rows))
            lines.append("")


def append_relative_tables(lines: list[str], relative: pd.DataFrame) -> None:
    for position in POSITIONS:
        for transform in TRANSFORMS:
            subset = relative[(relative.position == position) & (relative.transform == transform)]
            pivot = {int(layer): {row.pair_category: row for _, row in group.iterrows()} for layer, group in subset.groupby("layer")}
            rows = [[layer] + [relative_cell(pivot[layer][cat]) for cat in NON_BASELINE_CATEGORIES] for layer in sorted(pivot)]
            lines.extend([f"**{position} — {transform}**", ""])
            lines.extend(md_table(["Layer"] + [CATEGORY_LABELS[c] for c in NON_BASELINE_CATEGORIES], rows))
            lines.append("")


def append_regression_tables(lines: list[str], regression: pd.DataFrame) -> None:
    for position in POSITIONS:
        for transform in TRANSFORMS:
            subset = regression[(regression.position == position) & (regression.transform == transform)].sort_values("layer")
            rows = []
            for _, row in subset.iterrows():
                rows.append(
                    [
                        int(row.layer),
                        coef_cell(row, "intercept"),
                        coef_cell(row, "coef_same_context"),
                        coef_cell(row, "coef_same_number"),
                        coef_cell(row, "coef_abs_delta_value"),
                        f"{fmt(row.r2_descriptive)} [{fmt(row.r2_ci_low)}, {fmt(row.r2_ci_high)}]",
                        int(row.n_pair_rows),
                    ]
                )
            lines.extend([f"**{position} — {transform}**", ""])
            lines.extend(md_table(["Layer", "Intercept", "Same context", "Same number", "|Δvalue|", "R²", "n"], rows))
            lines.append("")


def probe_tables(lines: list[str], probes: pd.DataFrame) -> None:
    def metric_cell(row: pd.Series, metric: str) -> str:
        return f"{fmt(row[metric + '_mean'])} [{fmt(row[metric + '_ci_low'])}, {fmt(row[metric + '_ci_high'])}]"

    for position in POSITIONS:
        for split in ["held_out_templates", "held_out_interpolation_values"]:
            subset = probes[(probes.position == position) & (probes.split == split)]
            rows = []
            for layer in sorted(subset.layer.unique()):
                true = subset[(subset.layer == layer) & (subset.control == "true_labels")].iloc[0]
                shuffle = subset[(subset.layer == layer) & (subset.control == "shuffled_value_labels")].iloc[0]
                rows.append(
                    [
                        int(layer),
                        metric_cell(true, "r2"),
                        metric_cell(true, "mae"),
                        metric_cell(shuffle, "r2"),
                        metric_cell(shuffle, "mae"),
                        int(true.n_folds),
                    ]
                )
            lines.extend([f"**{position} — {split}**", ""])
            lines.extend(md_table(["Layer", "True R²", "True MAE", "Shuffled R²", "Shuffled MAE", "folds"], rows))
            lines.append("")


def probe_highlights(probes: pd.DataFrame, final_layer: int) -> list[list[object]]:
    rows = []
    for position in POSITIONS:
        for split in ["held_out_templates", "held_out_interpolation_values"]:
            subset = probes[(probes.position == position) & (probes.split == split)]
            true = subset[subset.control == "true_labels"]
            best_layer = int(true.loc[true.r2_mean.idxmax(), "layer"])
            for label, layer in [("layer 0", 0), ("best", best_layer), ("final", final_layer)]:
                true_row = true[true.layer == layer].iloc[0]
                shuffled = subset[(subset.control == "shuffled_value_labels") & (subset.layer == layer)].iloc[0]
                rows.append(
                    [
                        position,
                        split,
                        label,
                        layer,
                        fmt(true_row.r2_mean),
                        fmt(true_row.mae_mean),
                        fmt(shuffled.r2_mean),
                        fmt(shuffled.mae_mean),
                    ]
                )
    return rows


def write_report(
    run_dir: Path,
    config: dict,
    stimuli: pd.DataFrame,
    summary: pd.DataFrame,
    relative: pd.DataFrame,
    regression: pd.DataFrame,
    probes: pd.DataFrame,
    samples: int,
    seed: int,
) -> None:
    layers = sorted(summary.layer.astype(int).unique())
    selected, middle, final_quartile = selected_layers(layers)
    template_folds = template_fold_details(stimuli)
    value_folds = value_fold_details(stimuli)
    values = sorted(stimuli.value.astype(int).unique())
    target_mean = float(np.mean(values))
    target_std = float(np.std(values))
    all_single_token = bool(((stimuli.number_token_end - stimuli.number_token_start) == 0).all())
    prompt_final_tokens = sorted(stimuli.tokens.astype(str).str.split().str[-1].unique())

    lines = [
        "# Expanded Geometry and Probe Validation Report",
        "",
        "> Regenerated only from the completed Llama-3.2-1B CSV outputs. No model was loaded, no activations were re-extracted, and no 3B experiment was started.",
        "",
        "## Scope and run identity",
        "",
        f"- Model: `{config.get('model_id', config.get('model'))}` at revision `{config.get('resolved_revision', 'unknown')}` ({int(config.get('parameter_count', 0)):,} parameters).",
        f"- Hidden-state layers: {layers[0]}–{layers[-1]} ({len(layers)} states, with layer 0 the embedding output).",
        f"- Stimuli: {len(stimuli)} = 3 contexts × 5 templates × 15 values ({values[0]}–{values[-1]} by 10).",
        "- Positions: final token of the numeral and final token of the prompt.",
        "- Cosines: raw, global mean-centered, and global per-dimension standardized.",
        "- This is descriptive geometry and linear decodability, not causal evidence.",
        "",
        "## Uncertainty and dependence",
        "",
        f"Pairwise uncertainty uses {samples} crossed cluster-bootstrap replicates (seed {seed}). Each replicate independently resamples the 15 context-specific template clusters and the 15 numeric-value clusters, then weights a dyad by the multiplicities of both endpoints. Category-versus-random differences use the same replicate for both means. This preserves major template/value dependence and replaces the original independent-row bootstrap. `n` remains the number of pair rows, not an independent-sample count.",
        "",
        "Probe intervals are the existing fold-level bootstrap intervals: five held-out-template folds or two held-out-value folds. The two-fold intervals are intrinsically unstable and should not be read as precise sampling uncertainty.",
        "",
        "Machine-readable expanded results: [pairwise cluster summary](pairwise_cluster_summary.csv), [category minus random](pairwise_relative_to_random.csv), and [cluster-aware regressions](pairwise_regression_cluster_ci.csv).",
        "",
        "## Compact geometry views",
        "",
        f"The middle layer is {middle}. The final quartile is defined as the last ceil({len(layers)}/4) hidden states: layers {final_quartile[0]}–{final_quartile[-1]}. The compact selection is {', '.join(map(str, selected))}; the final layer is included in the final quartile and shown explicitly as the endpoint.",
        "",
    ]
    append_geometry_tables(lines, summary, selected, "Layer 0, middle, final quartile, and final layer")

    lines.extend(
        [
            "## Category differences from the random baseline",
            "",
            "Cells are category mean minus random-baseline mean [crossed-cluster-bootstrap 95% CI]. A positive value means greater cosine than the sampled cross-context/different-value baseline.",
            "",
        ]
    )
    append_relative_tables(lines, relative)

    lines.extend(
        [
            "## Pairwise regression",
            "",
            "For every position, layer, and correction, cosine was regressed on same-context, same-number, and absolute numeric distance. Coefficients and descriptive in-sample R² are point estimates from all pair rows; brackets are crossed template/value cluster-bootstrap 95% CIs. The distance coefficient is per one raw numeric unit. These are descriptive associations among non-independent dyads, not causal effects.",
            "",
        ]
    )
    append_regression_tables(lines, regression)

    lines.extend(
        [
            "## Probe highlights: layer 0, best, and final",
            "",
            "MAE is in the globally z-scored target units used by the completed run. Negative R² values are retained without clipping.",
            "",
        ]
    )
    lines.extend(
        md_table(
            ["Position", "Split", "Checkpoint", "Layer", "True R²", "True MAE", "Shuffled R²", "Shuffled MAE"],
            probe_highlights(probes, layers[-1]),
        )
    )
    lines.extend(["", "## Probe results at every layer", "", "Cells are mean [existing fold-bootstrap 95% CI].", ""])
    probe_tables(lines, probes)

    lines.extend(["## Exact held-out splits", "", "### Held-out-template folds", ""])
    lines.append("Each fold holds out one template index simultaneously in all three contexts. Every fold trains on all 15 values and the other four template indices (n=180), then tests all 15 values in the three held-out literal templates (n=45). Thus this split holds out wording, but not numeral token identities.")
    lines.append("")
    for fold in template_folds:
        lines.extend([f"**Fold {fold['fold']}**", "", f"- Train template indices: {', '.join(fold['train_templates'])}; train values: {', '.join(map(str, fold['train_values']))}; n={fold['n_train']}.", f"- Test values: {', '.join(map(str, fold['test_values']))}; n={fold['n_test']}.", "- Test literal templates:"])
        lines.extend([f"  - {template}" for template in fold["test_templates"]])
        lines.append("")

    lines.extend(["### Held-out-interpolation-value folds", ""])
    lines.append("Endpoints 40 and 180 are never test values; they remain in training. All 15 context/template combinations occur in both train and test, so this split tests interpolation to unseen values within familiar templates rather than template transfer.")
    lines.append("")
    value_rows = []
    for fold in value_folds:
        value_rows.append([fold["fold"], ", ".join(map(str, fold["train_values"])), ", ".join(map(str, fold["test_values"])), fold["n_train"], fold["n_test"], "all 15", "all 15"])
    lines.extend(md_table(["Fold", "Train values", "Test values", "n train", "n test", "Train templates", "Test templates"], value_rows))

    lines.extend(
        [
            "",
            "## Preprocessing audit",
            "",
            "| Component | What the completed run did | Train-only? | Consequence |",
            "|---|---|---|---|",
            "| Probe feature scaling | `StandardScaler` is the first step of a scikit-learn pipeline fitted separately inside every fold, before ridge prediction. | Yes | Test activation means/variances do not enter feature scaling. |",
            f"| Probe target z-scoring | Numeric labels were transformed once using the full 15-value set (mean {target_mean:.1f}, population SD {target_std:.3f}) before folds were formed. | **No** | This leaks test-label distribution into the target scale. R² is invariant to this common affine rescaling, but MAE is reported on a globally defined scale and is not a strictly train-only held-out MAE. |",
            "| Ridge fitting | Ridge (alpha=10) is fitted only on each training fold; alpha was fixed rather than selected on the test data. | Yes | No test-feature fitting or test-driven hyperparameter tuning is visible. |",
            "| Pairwise mean-centering | Per-layer vectors were centered using the mean over all 225 stimuli before pairwise cosines. | No split applies | Appropriate only as transductive descriptive geometry; it must not be interpreted as held-out predictive preprocessing. |",
            "| Pairwise standardization | Per-layer, per-dimension means and SDs were computed over all 225 stimuli before cosine. | No split applies | Also transductive descriptive geometry. A future predictive use would need fold-specific fitting. |",
            "| Pairwise regression | Binary same-context/same-number indicators and raw absolute value difference were used without a learned transformation. | Not applicable | Coefficients are descriptive; dependence is handled here through cluster-bootstrap intervals. |",
            "| Shuffled control | A label permutation was made before fold fitting for each analysis condition, while the scaler/ridge pipeline was still fitted within folds. | Mostly | Valid as a control for the recorded analysis, though multiple permutations would give a more stable null distribution than one permutation per condition. |",
            "",
            "The requested audit therefore does **not** support a blanket statement that all preprocessing was train-only. Probe feature z-scoring was train-only; target z-scoring was not. Pairwise corrections were global because that analysis had no train/test split.",
            "",
            "## Diagnostics",
            "",
            "### Why numeral-final held-out-template R² is 1.0 at layer 0",
            "",
            f"All {len(stimuli)} numerals are one token (`number_token_start == number_token_end`: {all_single_token}). Layer 0 at the numeral position is the token-embedding output, before contextual transformer layers. The held-out-template folds retain every value—and therefore every numeral token identity—in training. The pairwise CSV independently shows layer-0 cosine of 1.0 for same-number pairs across templates/contexts under all three corrections. The near-perfect template-fold probe (R²=1.000, MAE=0.000147) can therefore be attributed largely to reuse of the same numeral-token embeddings, not transfer of a context-invariant numerical computation to unseen number identities. This design cannot distinguish an identity lookup from numerical structure in the embedding geometry; alternate spellings, multi-token numerals, or number-identity-held-out tests would be needed.",
            "",
            "### Why prompt-final template transfer fails while value interpolation succeeds",
            "",
            f"Every prompt ends in the same final token ({', '.join(repr(x) for x in prompt_final_tokens)}). At layer 0 its embedding is therefore constant across stimuli, giving the null mean predictor: R²=0 for both splits (the held-out-value value is numerical roundoff near zero). The often-quoted pair of scores compares maxima at different layers: held-out-template's best is 0 at layer 0, whereas held-out-value's best is 0.938 at layer 6. At layer 6 itself, held-out-template R² is −2.272, so its predictions are worse than the test-fold mean baseline.",
            "",
            "From layer 1 onward, the prompt-final state has incorporated preceding tokens. The held-out-value split exposes every literal template during training and only withholds alternating interior values, so a template-specific mapping can interpolate well. The held-out-template split withholds the literal wording and tests whether one linear map transfers to unseen surface forms; it does not. This pattern is consistent with template-specific offsets/directions, covariate shift, or an inadequately invariant linear probe. The current aggregate CSVs do not isolate which mechanism is responsible. It is **not** evidence for context-dependent construction, because there is no decision task, causal intervention, alternate-token control, or fold-level error decomposition in the saved outputs.",
            "",
            "## Plots",
            "",
            "The following plots were regenerated from `pairwise_measurements.csv` with the crossed template/value cluster bootstrap:",
            "",
            "![Cluster-aware numeral-final geometry](cluster_geometry_numeral_final.png)",
            "",
            "![Cluster-aware prompt-final geometry](cluster_geometry_prompt_final.png)",
            "",
            "Existing probe plots (their intervals resample held-out folds and negative R² is not clipped):",
            "",
            "- [Numeral-final, held-out templates](probe_r2_numeral_final_held_out_templates.png)",
            "- [Numeral-final, held-out values](probe_r2_numeral_final_held_out_interpolation_values.png)",
            "- [Prompt-final, held-out templates](probe_r2_prompt_final_held_out_templates.png)",
            "- [Prompt-final, held-out values](probe_r2_prompt_final_held_out_interpolation_values.png)",
            "",
            "## Evidence boundary",
            "",
            "| Class | Statement |",
            "|---|---|",
            "| Direct observation | The saved run has 225 stimuli, 17 hidden-state layers (0–16), two positions, three cosine corrections, and complete per-layer probe rows. |",
            "| Direct observation | Every numeral is a single token; every prompt-final token is a period; template folds reuse all 15 numeral identities; value folds reuse all 15 literal templates. |",
            "| Direct observation | Numeral-final layer-0 held-out-template R² is 1.000; prompt-final's maximum held-out-template R² is 0 at layer 0; prompt-final held-out-value R² peaks at 0.938 at layer 6, where template-held-out R² is −2.272. |",
            "| Direct observation | Feature scaling is fitted within probe training folds, while target z-scoring is global and pairwise centering/standardization use all stimuli. |",
            "| Conclusion supported by current evidence | Numeral magnitude is linearly decodable under these particular splits, with especially strong within-template value interpolation at prompt-final layers. |",
            "| Conclusion supported by current evidence | The prompt-final linear map does not generalize to the held-out literal templates in this stimulus set; negative R² means it underperforms the fold mean baseline. |",
            "| Conclusion supported by current evidence | The layer-0 numeral template result is largely explained by unchanged numeral token identity across train/test templates. |",
            "| Interpretation not yet supported | The model constructs numerical representations in a context-dependent or task-dependent manner. |",
            "| Interpretation not yet supported | High cosine or probe R² demonstrates causal use of magnitude by the model. |",
            "| Interpretation not yet supported | Template-transfer failure identifies a specific mechanism (context gating, attention head, MLP, or loss of numerical information). |",
            "| Interpretation not yet supported | The uncertainty estimates generalize beyond the 15 hand-written templates and 15 values; clusters remain few, especially the two value folds. |",
            "",
            "## Full per-layer cosine estimates",
            "",
            "This appendix satisfies the complete factorial reporting requirement: every layer × extraction position × correction × pair category is shown as mean [crossed-cluster-bootstrap 95% CI]; n.",
            "",
        ]
    )
    append_geometry_tables(lines, summary, layers, "All layers")
    (run_dir / "geometry_probe_validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    config, stimuli, pairs, old_regression, probes = read_inputs(run_dir)
    samples = int(args.bootstrap_samples if args.bootstrap_samples is not None else config.get("bootstrap_samples", 500))
    seed = int(args.seed if args.seed is not None else config.get("seed", 13))
    pairs, n_templates, n_values = add_cluster_ids(pairs, stimuli)
    summary, relative, regression = cluster_bootstrap_analysis(pairs, samples, seed, n_templates, n_values)

    merged = regression.merge(
        old_regression,
        on=["position", "layer", "transform"],
        suffixes=("", "_saved"),
        validate="one_to_one",
    )
    for name in ["intercept", "coef_same_context", "coef_same_number", "coef_abs_delta_value", "r2_descriptive"]:
        if not np.allclose(merged[name], merged[name + "_saved"], atol=1e-10, rtol=1e-8):
            raise ValueError(f"Recomputed regression point estimates do not match saved {name}")

    summary.to_csv(run_dir / "pairwise_cluster_summary.csv", index=False)
    relative.to_csv(run_dir / "pairwise_relative_to_random.csv", index=False)
    regression.to_csv(run_dir / "pairwise_regression_cluster_ci.csv", index=False)
    geometry_plots(summary, relative, run_dir)
    write_report(run_dir, config, stimuli, summary, relative, regression, probes, samples, seed)
    print(f"Expanded report regenerated from CSVs only: {run_dir / 'geometry_probe_validation_report.md'}")


if __name__ == "__main__":
    main()
