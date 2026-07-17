"""
Geometry and linear-probe validation for numerical-context representations.

This experiment is deliberately separate from mechanistic patching. It tests
whether late-layer cosine convergence persists after simple anisotropy controls
and whether numeric magnitude is linearly decodable from residual-stream states.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import transformers
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL = "meta-llama/Llama-3.2-1B"
SMOKE_TEST_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"
POSITIONS = ("numeral_final", "prompt_final")
TRANSFORMS = ("raw", "centered", "standardized")
PAIR_CATEGORIES = (
    "same_context_different_number",
    "same_number_different_context",
    "same_context_same_number_different_template",
    "random_baseline",
)


@dataclass(frozen=True)
class ExperimentConfig:
    model: str = DEFAULT_MODEL
    output_root: str = "results/exp_geometry_probe_validation"
    values_start: int = 40
    values_stop: int = 180
    values_step: int = 10
    batch_size: int = 4
    seed: int = 13
    ridge_alpha: float = 10.0
    bootstrap_samples: int = 500
    max_pairs_per_category: int = 20000
    smoke_test: bool = False
    save_activations: bool = False
    device: str = "auto"
    validate_load_only: bool = False


@dataclass(frozen=True)
class Stimulus:
    stimulus_id: str
    value: int
    context: str
    template_id: str
    template_index: int
    prompt: str


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Full-run model ID (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument("--output-root", default="results/exp_geometry_probe_validation")
    parser.add_argument("--values-start", type=int, default=40)
    parser.add_argument("--values-stop", type=int, default=180)
    parser.add_argument("--values-step", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--max-pairs-per-category", type=int, default=20000)
    parser.add_argument("--save-activations", action="store_true")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=f"Use {SMOKE_TEST_MODEL} and a small CPU-friendly stimulus set.",
    )
    parser.add_argument(
        "--validate-load-only",
        action="store_true",
        help="Load the checkpoint, report runtime metadata, then exit without running the experiment.",
    )
    args = parser.parse_args()
    cfg = ExperimentConfig(**vars(args))
    if cfg.smoke_test:
        cfg = ExperimentConfig(
            **{
                **asdict(cfg),
                "model": SMOKE_TEST_MODEL,
                "values_start": cfg.values_start,
                "values_stop": min(cfg.values_stop, cfg.values_start + cfg.values_step * 2),
                "bootstrap_samples": min(cfg.bootstrap_samples, 50),
                "max_pairs_per_category": min(cfg.max_pairs_per_category, 2000),
            }
        )
    elif "tiny-random" in cfg.model.lower() or "testing" in cfg.model.lower():
        raise ValueError(
            "A full run requires a real pretrained checkpoint. "
            f"Use --smoke-test to run {SMOKE_TEST_MODEL}."
        )
    return cfg


def set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def choose_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        return torch.device("cuda")
    if name == "mps":
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def context_templates() -> Dict[str, List[str]]:
    return {
        "medical_hr": [
            "The patient's heart rate was {value} beats per minute.",
            "Clinical monitoring recorded a heart rate of {value} bpm.",
            "The nurse documented the patient's pulse as {value} bpm.",
            "During triage, the heart-rate reading was {value} beats per minute.",
            "The bedside monitor showed HR at {value} bpm.",
        ],
        "travel_duration": [
            "The flight duration was {value} hours.",
            "The trip itinerary listed the travel time as {value} hours.",
            "The route took {value} hours from departure to arrival.",
            "The long-distance journey lasted {value} hours.",
            "The schedule recorded {value} hours of total travel time.",
        ],
        "object_count": [
            "The storage room contained {value} boxes.",
            "The inventory count showed {value} objects.",
            "The collection included {value} items on the shelf.",
            "The warehouse log listed {value} packages.",
            "The basket held {value} small objects.",
        ],
    }


def build_stimuli(cfg: ExperimentConfig) -> List[Stimulus]:
    values = list(range(cfg.values_start, cfg.values_stop + 1, cfg.values_step))
    if cfg.smoke_test:
        values = values[:3]

    rows: List[Stimulus] = []
    for context, templates in context_templates().items():
        selected_templates = templates[:2] if cfg.smoke_test else templates
        for template_index, template in enumerate(selected_templates):
            template_id = f"T{template_index + 1}"
            for value in values:
                stim_id = f"{context}_{template_id}_{value}"
                rows.append(
                    Stimulus(
                        stimulus_id=stim_id,
                        value=value,
                        context=context,
                        template_id=template_id,
                        template_index=template_index,
                        prompt=template.format(value=value),
                    )
                )
    return rows


def clean_token(token: str) -> str:
    cleaned = token.replace("\u0120", "").replace("\u2581", "").replace("\u010a", "")
    cleaned = cleaned.replace("<0x0A>", "").strip()
    return cleaned


def find_number_span(tokenizer, input_ids: Sequence[int], value: int) -> List[int]:
    number = str(value)
    target_variants = [
        tokenizer.encode(number, add_special_tokens=False),
        tokenizer.encode(" " + number, add_special_tokens=False),
    ]
    ids = list(input_ids)
    for target in target_variants:
        if not target:
            continue
        for start in range(0, len(ids) - len(target) + 1):
            if ids[start : start + len(target)] == target:
                return list(range(start, start + len(target)))

    tokens = tokenizer.convert_ids_to_tokens(ids)
    for start in range(len(tokens)):
        acc = ""
        span: List[int] = []
        for end in range(start, len(tokens)):
            acc += clean_token(tokens[end])
            span.append(end)
            if acc == number:
                return span
            if not number.startswith(acc):
                break
    raise ValueError(f"Could not find numeral {number!r} in tokens={tokens}")


def batched(items: Sequence[Stimulus], batch_size: int) -> Iterable[List[Stimulus]]:
    for start in range(0, len(items), batch_size):
        yield list(items[start : start + batch_size])


def resolved_revision(model, tokenizer) -> str:
    return str(
        getattr(model.config, "_commit_hash", None)
        or tokenizer.init_kwargs.get("_commit_hash")
        or "unknown"
    )


def load_model_and_tokenizer(cfg: ExperimentConfig):
    device = choose_device(cfg.device)
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32
    try:
        tokenizer = AutoTokenizer.from_pretrained(cfg.model, token=True)
        model = AutoModelForCausalLM.from_pretrained(cfg.model, dtype=dtype, token=True).to(device)
    except OSError as exc:
        raise RuntimeError(
            f"Could not load {cfg.model!r}. This experiment never falls back to a tiny or random model. "
            "For Meta Llama gated checkpoints, accept the model license on Hugging Face and run "
            "`huggingface-cli login` with a token that has access; then retry. "
            f"Original error: {exc}"
        ) from exc
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    metadata = {
        "model_id": cfg.model,
        "resolved_revision": resolved_revision(model, tokenizer),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "dtype": str(dtype).replace("torch.", ""),
        "device": str(device),
        "transformers_version": transformers.__version__,
        "seed": cfg.seed,
        "smoke_test": cfg.smoke_test,
    }
    return tokenizer, model, device, metadata

def extract_activations(cfg: ExperimentConfig, stimuli: List[Stimulus], out_dir: Path):
    tokenizer, model, device, metadata = load_model_and_tokenizer(cfg)
    num_layers = None
    activations: Dict[str, List[np.ndarray]] = {pos: [] for pos in POSITIONS}
    meta_rows = []

    for batch in batched(stimuli, cfg.batch_size):
        prompts = [s.prompt for s in batch]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            outputs = model(**encoded, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        if num_layers is None:
            num_layers = len(hidden_states)

        input_ids = encoded["input_ids"].detach().cpu().tolist()
        attention_mask = encoded["attention_mask"].detach().cpu().numpy()
        for row_idx, stim in enumerate(batch):
            ids = input_ids[row_idx]
            real_len = int(attention_mask[row_idx].sum())
            unpadded_ids = ids[:real_len]
            span = find_number_span(tokenizer, unpadded_ids, stim.value)
            numeral_final = span[-1]
            prompt_final = real_len - 1
            meta_rows.append(
                {
                    **asdict(stim),
                    "number_token_start": span[0],
                    "number_token_end": span[-1],
                    "number_token_span": " ".join(str(i) for i in span),
                    "prompt_final_token": prompt_final,
                    "tokens": " ".join(tokenizer.convert_ids_to_tokens(unpadded_ids)),
                }
            )

            for position_name, token_pos in (
                ("numeral_final", numeral_final),
                ("prompt_final", prompt_final),
            ):
                layer_vecs = [
                    h[row_idx, token_pos].detach().float().cpu().numpy()
                    for h in hidden_states
                ]
                activations[position_name].append(np.stack(layer_vecs, axis=0))

    stacked = {pos: np.stack(vecs, axis=0) for pos, vecs in activations.items()}
    write_csv(out_dir / "stimuli.csv", meta_rows)
    if cfg.save_activations:
        np.savez_compressed(out_dir / "activations.npz", **stacked)
    return stacked, meta_rows, int(num_layers or 0), metadata


def write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def transform_vectors(x: np.ndarray, mode: str) -> np.ndarray:
    if mode == "raw":
        return x
    mean = x.mean(axis=0, keepdims=True)
    centered = x - mean
    if mode == "centered":
        return centered
    std = x.std(axis=0, keepdims=True)
    return centered / np.where(std < 1e-6, 1.0, std)


def cosine_matrix(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    safe = x / np.where(norms < 1e-8, 1.0, norms)
    return safe @ safe.T


def pair_category(a: Dict, b: Dict) -> str:
    same_context = a["context"] == b["context"]
    same_value = int(a["value"]) == int(b["value"])
    same_template = a["template_id"] == b["template_id"]
    if same_context and same_value and not same_template:
        return "same_context_same_number_different_template"
    if same_context and not same_value:
        return "same_context_different_number"
    if same_value and not same_context:
        return "same_number_different_context"
    return "random_baseline"


def compute_pairwise_metrics(
    cfg: ExperimentConfig,
    activations: Dict[str, np.ndarray],
    meta_rows: List[Dict],
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    rng = np.random.default_rng(cfg.seed)
    pair_rows: List[Dict] = []
    regression_rows: List[Dict] = []
    delta_rows: List[Dict] = []
    n = len(meta_rows)

    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    categorized: Dict[str, List[Tuple[int, int]]] = {cat: [] for cat in PAIR_CATEGORIES}
    for i, j in all_pairs:
        categorized[pair_category(meta_rows[i], meta_rows[j])].append((i, j))

    sampled_pairs = []
    for cat, pairs in categorized.items():
        if len(pairs) > cfg.max_pairs_per_category:
            idx = rng.choice(len(pairs), size=cfg.max_pairs_per_category, replace=False)
            pairs = [pairs[k] for k in idx]
        sampled_pairs.extend((cat, i, j) for i, j in pairs)
        deltas = [abs(int(meta_rows[i]["value"]) - int(meta_rows[j]["value"])) for i, j in pairs]
        if deltas:
            delta_rows.append(
                {
                    "pair_category": cat,
                    "n_pairs": len(deltas),
                    "abs_delta_mean": float(np.mean(deltas)),
                    "abs_delta_median": float(np.median(deltas)),
                    "abs_delta_min": int(np.min(deltas)),
                    "abs_delta_max": int(np.max(deltas)),
                }
            )

    for position in POSITIONS:
        x_all = activations[position]
        num_layers = x_all.shape[1]
        for layer in range(num_layers):
            layer_x = x_all[:, layer, :]
            for transform in TRANSFORMS:
                sim = cosine_matrix(transform_vectors(layer_x, transform))
                y_reg, features = [], []
                for cat, i, j in sampled_pairs:
                    a = meta_rows[i]
                    b = meta_rows[j]
                    abs_delta = abs(int(a["value"]) - int(b["value"]))
                    similarity = float(sim[i, j])
                    pair_rows.append(
                        {
                            "position": position,
                            "layer": layer,
                            "transform": transform,
                            "pair_category": cat,
                            "stimulus_id_a": a["stimulus_id"],
                            "stimulus_id_b": b["stimulus_id"],
                            "context_a": a["context"],
                            "context_b": b["context"],
                            "value_a": int(a["value"]),
                            "value_b": int(b["value"]),
                            "abs_delta_value": abs_delta,
                            "same_context": int(a["context"] == b["context"]),
                            "same_number": int(int(a["value"]) == int(b["value"])),
                            "similarity": similarity,
                        }
                    )
                    y_reg.append(similarity)
                    features.append(
                        [
                            int(a["context"] == b["context"]),
                            int(int(a["value"]) == int(b["value"])),
                            abs_delta,
                        ]
                    )
                if len(y_reg) >= 4:
                    lr = LinearRegression().fit(np.asarray(features), np.asarray(y_reg))
                    regression_rows.append(
                        {
                            "position": position,
                            "layer": layer,
                            "transform": transform,
                            "intercept": float(lr.intercept_),
                            "coef_same_context": float(lr.coef_[0]),
                            "coef_same_number": float(lr.coef_[1]),
                            "coef_abs_delta_value": float(lr.coef_[2]),
                            "r2_descriptive": float(lr.score(np.asarray(features), np.asarray(y_reg))),
                            "n_pairs": len(y_reg),
                        }
                    )
    return pair_rows, regression_rows, delta_rows


def summarize_pairs(pair_rows: List[Dict]) -> List[Dict]:
    grouped: Dict[Tuple[str, int, str, str], List[float]] = {}
    for row in pair_rows:
        key = (row["position"], int(row["layer"]), row["transform"], row["pair_category"])
        grouped.setdefault(key, []).append(float(row["similarity"]))
    summary = []
    for (position, layer, transform, category), vals in sorted(grouped.items()):
        arr = np.asarray(vals)
        summary.append(
            {
                "position": position,
                "layer": layer,
                "transform": transform,
                "pair_category": category,
                "n_pairs": len(vals),
                "mean_similarity": float(arr.mean()),
                "std_similarity": float(arr.std(ddof=1)) if len(vals) > 1 else 0.0,
                "median_similarity": float(np.median(arr)),
            }
        )
    return summary


def bootstrap_ci(vals: Sequence[float], rng: np.random.Generator, samples: int) -> Tuple[float, float]:
    arr = np.asarray(vals, dtype=float)
    if len(arr) == 0:
        return float("nan"), float("nan")
    if len(arr) == 1 or samples <= 0:
        return float(arr[0]), float(arr[0])
    means = [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(samples)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def plot_pair_lines(cfg: ExperimentConfig, pair_rows: List[Dict], out_dir: Path) -> None:
    rng = np.random.default_rng(cfg.seed)
    for position in POSITIONS:
        for transform in TRANSFORMS:
            fig, ax = plt.subplots(figsize=(11, 6))
            for category in PAIR_CATEGORIES:
                by_layer: Dict[int, List[float]] = {}
                for row in pair_rows:
                    if row["position"] == position and row["transform"] == transform and row["pair_category"] == category:
                        by_layer.setdefault(int(row["layer"]), []).append(float(row["similarity"]))
                layers = sorted(by_layer)
                if not layers:
                    continue
                means = [float(np.mean(by_layer[layer])) for layer in layers]
                cis = [bootstrap_ci(by_layer[layer], rng, cfg.bootstrap_samples) for layer in layers]
                lo = [x[0] for x in cis]
                hi = [x[1] for x in cis]
                ax.plot(layers, means, marker="o", linewidth=2, label=category)
                ax.fill_between(layers, lo, hi, alpha=0.15)
            ax.set_title(f"{position}: {transform} cosine by pair category")
            ax.set_xlabel("Layer")
            ax.set_ylabel("Cosine similarity")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.25)
            fig.tight_layout()
            fig.savefig(out_dir / f"pair_lines_{position}_{transform}.png", dpi=180)
            plt.close(fig)


def plot_transform_comparison(pair_summary: List[Dict], out_dir: Path) -> None:
    for position in POSITIONS:
        fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, sharey=True)
        for ax, category in zip(axes.flatten(), PAIR_CATEGORIES):
            for transform in TRANSFORMS:
                rows = [
                    row
                    for row in pair_summary
                    if row["position"] == position
                    and row["pair_category"] == category
                    and row["transform"] == transform
                ]
                rows = sorted(rows, key=lambda r: int(r["layer"]))
                if not rows:
                    continue
                ax.plot(
                    [int(r["layer"]) for r in rows],
                    [float(r["mean_similarity"]) for r in rows],
                    marker="o",
                    linewidth=2,
                    label=transform,
                )
            ax.set_title(category)
            ax.grid(alpha=0.25)
        axes[1, 0].set_xlabel("Layer")
        axes[1, 1].set_xlabel("Layer")
        axes[0, 0].set_ylabel("Mean cosine")
        axes[1, 0].set_ylabel("Mean cosine")
        axes[0, 0].legend(fontsize=8)
        fig.suptitle(f"Raw vs centered vs standardized cosine: {position}")
        fig.tight_layout()
        fig.savefig(out_dir / f"transform_comparison_{position}.png", dpi=180)
        plt.close(fig)


def interpolation_value_folds(values: Sequence[int]) -> List[List[int]]:
    vals = sorted(set(values))
    inner = vals[1:-1]
    if not inner:
        return [vals[-1:]]
    return [inner[::2], inner[1::2] or inner[:1]]


def metric_ci(vals: Sequence[float], rng: np.random.Generator, samples: int) -> Tuple[float, float]:
    return bootstrap_ci(vals, rng, samples)


def run_probe_analysis(
    cfg: ExperimentConfig,
    activations: Dict[str, np.ndarray],
    meta_rows: List[Dict],
) -> List[Dict]:
    rng = np.random.default_rng(cfg.seed)
    y_raw = np.asarray([int(r["value"]) for r in meta_rows], dtype=float)
    y = (y_raw - y_raw.mean()) / (y_raw.std() + 1e-8)
    template_indices = np.asarray([int(r["template_index"]) for r in meta_rows])
    values = np.asarray([int(r["value"]) for r in meta_rows])
    unique_templates = sorted(set(template_indices.tolist()))
    template_folds = [np.where(template_indices == t)[0] for t in unique_templates]
    value_folds = [np.where(np.isin(values, fold))[0] for fold in interpolation_value_folds(values.tolist())]

    rows: List[Dict] = []
    for position in POSITIONS:
        x_all = activations[position]
        num_layers = x_all.shape[1]
        for layer in range(num_layers):
            x = x_all[:, layer, :]
            for split_name, folds in (
                ("held_out_templates", template_folds),
                ("held_out_interpolation_values", value_folds),
            ):
                for control_name, labels in (
                    ("true_labels", y),
                    ("shuffled_value_labels", rng.permutation(y)),
                ):
                    fold_r2, fold_mae = [], []
                    for test_idx in folds:
                        if len(test_idx) == 0:
                            continue
                        train_mask = np.ones(len(y), dtype=bool)
                        train_mask[test_idx] = False
                        if len(np.unique(labels[train_mask])) < 2:
                            continue
                        model = make_pipeline(
                            StandardScaler(),
                            Ridge(alpha=cfg.ridge_alpha, random_state=cfg.seed),
                        )
                        model.fit(x[train_mask], labels[train_mask])
                        pred = model.predict(x[test_idx])
                        fold_r2.append(r2_score(labels[test_idx], pred))
                        fold_mae.append(mean_absolute_error(labels[test_idx], pred))
                    if fold_r2:
                        r2_lo, r2_hi = metric_ci(fold_r2, rng, cfg.bootstrap_samples)
                        mae_lo, mae_hi = metric_ci(fold_mae, rng, cfg.bootstrap_samples)
                        rows.append(
                            {
                                "position": position,
                                "layer": layer,
                                "split": split_name,
                                "control": control_name,
                                "r2_mean": float(np.mean(fold_r2)),
                                "r2_ci_low": r2_lo,
                                "r2_ci_high": r2_hi,
                                "mae_mean": float(np.mean(fold_mae)),
                                "mae_ci_low": mae_lo,
                                "mae_ci_high": mae_hi,
                                "n_folds": len(fold_r2),
                            }
                        )
    return rows


def plot_probe(probe_rows: List[Dict], out_dir: Path) -> None:
    for position in POSITIONS:
        for split in ("held_out_templates", "held_out_interpolation_values"):
            fig, ax = plt.subplots(figsize=(10, 5.5))
            for control in ("true_labels", "shuffled_value_labels"):
                rows = [
                    r
                    for r in probe_rows
                    if r["position"] == position and r["split"] == split and r["control"] == control
                ]
                rows = sorted(rows, key=lambda r: int(r["layer"]))
                if not rows:
                    continue
                layers = [int(r["layer"]) for r in rows]
                means = [float(r["r2_mean"]) for r in rows]
                lo = [float(r["r2_ci_low"]) for r in rows]
                hi = [float(r["r2_ci_high"]) for r in rows]
                ax.plot(layers, means, marker="o", linewidth=2, label=control)
                ax.fill_between(layers, lo, hi, alpha=0.15)
            ax.axhline(0, color="black", linewidth=1, alpha=0.4)
            ax.set_title(f"Ridge probe R2: {position}, {split}")
            ax.set_xlabel("Layer")
            ax.set_ylabel("R2 predicting z-scored numeric value")
            ax.legend()
            ax.grid(alpha=0.25)
            fig.tight_layout()
            fig.savefig(out_dir / f"probe_r2_{position}_{split}.png", dpi=180)
            plt.close(fig)


def write_report(
    cfg: ExperimentConfig,
    run_metadata: Dict,
    out_dir: Path,
    pair_summary: List[Dict],
    probe_rows: List[Dict],
    delta_rows: List[Dict],
) -> None:
    if not cfg.smoke_test and any(marker in cfg.model.lower() for marker in ("tiny-random", "testing")):
        raise RuntimeError("Refusing to label a tiny/testing checkpoint report as a full run.")
    def best_probe(position: str, split: str) -> Dict:
        rows = [
            r
            for r in probe_rows
            if r["position"] == position and r["split"] == split and r["control"] == "true_labels"
        ]
        return max(rows, key=lambda r: float(r["r2_mean"])) if rows else {}

    lines = [
        "# Geometry and Probe Validation Report",
        "",
        "This report summarizes a descriptive geometry-and-probing validation study. "
        "It does not use activation patching, logit lens, attention-head attribution, "
        "MLP attribution, or normalization ablations.",
        "",
        "## Configuration",
        "",
        "```json",
        json.dumps({**asdict(cfg), **run_metadata}, indent=2),
        "```",
        "",
        "## What Was Measured",
        "",
        "- Residual-stream hidden states were extracted at every model layer.",
        "- Two positions were analyzed: final numeral token and prompt-final token.",
        "- Pairwise cosine similarity was computed as raw, layer-wise mean-centered, "
        "and layer-wise per-dimension standardized cosine.",
        "- Ridge probes predicted z-scored numeric value under held-out-template and "
        "held-out-interpolation-value splits.",
        "",
        "## Pair Balance Check",
        "",
    ]
    for row in delta_rows:
        lines.append(
            f"- {row['pair_category']}: n={row['n_pairs']}, "
            f"abs_delta mean={float(row['abs_delta_mean']):.2f}, "
            f"median={float(row['abs_delta_median']):.2f}, "
            f"range=[{row['abs_delta_min']}, {row['abs_delta_max']}]."
        )

    lines.extend(["", "## Probe Highlights", ""])
    for position in POSITIONS:
        for split in ("held_out_templates", "held_out_interpolation_values"):
            row = best_probe(position, split)
            if row:
                lines.append(
                    f"- {position}, {split}: best true-label R2={float(row['r2_mean']):.3f} "
                    f"at layer {row['layer']} with MAE={float(row['mae_mean']):.3f}."
                )

    lines.extend(
        [
            "",
            "## Interpretation Guardrails",
            "",
            "- High raw cosine alone may reflect anisotropy or shared-template effects.",
            "- High corrected cosine does not by itself show that numerical information is absent.",
            "- Above-control probe performance means magnitude is linearly decodable; it does not show causal use.",
            "- A reduction in probe performance is not proof that the model has forgotten the number.",
            "- This experiment does not test task-dependent construction, because there is no decision task or causal intervention.",
            "",
            "## Output Files",
            "",
            "- `stimuli.csv`: generated prompts with validated numeral spans.",
            "- `pairwise_measurements.csv`: sampled pairwise cosine rows.",
            "- `pairwise_summary.csv`: compact layer/category summaries.",
            "- `pairwise_regression.csv`: descriptive regressions of similarity on same-context, same-number, and absolute numeric distance.",
            "- `probe_results.csv`: ridge probe metrics and controls.",
            "- `*.png`: layer-wise geometry and probe plots.",
        ]
    )
    (out_dir / "geometry_probe_validation_report.md").write_text("\n".join(lines), encoding="utf-8")


def make_output_dir(cfg: ExperimentConfig) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_slug = cfg.model.replace("/", "_").replace("\\", "_")
    suffix = "smoke" if cfg.smoke_test else "full"
    out_dir = Path(cfg.output_root) / f"{stamp}_{model_slug}_{suffix}"
    out_dir.mkdir(parents=True, exist_ok=False)
    return out_dir


def main() -> None:
    cfg = parse_args()
    set_deterministic(cfg.seed)
    if cfg.validate_load_only:
        _tokenizer, _model, _device, metadata = load_model_and_tokenizer(cfg)
        print(json.dumps({**asdict(cfg), **metadata}, indent=2))
        return

    out_dir = make_output_dir(cfg)
    stimuli = build_stimuli(cfg)
    activations, meta_rows, _num_layers, metadata = extract_activations(cfg, stimuli, out_dir)
    run_config = {**asdict(cfg), **metadata}
    (out_dir / "config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    pair_rows, regression_rows, delta_rows = compute_pairwise_metrics(cfg, activations, meta_rows)
    pair_summary = summarize_pairs(pair_rows)
    probe_rows = run_probe_analysis(cfg, activations, meta_rows)

    write_csv(out_dir / "pairwise_measurements.csv", pair_rows)
    write_csv(out_dir / "pairwise_summary.csv", pair_summary)
    write_csv(out_dir / "pairwise_regression.csv", regression_rows)
    write_csv(out_dir / "pair_abs_delta_distributions.csv", delta_rows)
    write_csv(out_dir / "probe_results.csv", probe_rows)
    plot_pair_lines(cfg, pair_rows, out_dir)
    plot_transform_comparison(pair_summary, out_dir)
    plot_probe(probe_rows, out_dir)
    write_report(cfg, metadata, out_dir, pair_summary, probe_rows, delta_rows)

    print(f"Done. Outputs written to: {out_dir}")


if __name__ == "__main__":
    main()

