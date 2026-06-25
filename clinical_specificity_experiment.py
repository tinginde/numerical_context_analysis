"""
Clinical Specificity Experiment
===============================

Test whether decreasing clinical specificity makes the hidden-state
representation of "24" more similar to a non-clinical baseline.

Primary baseline: "The value is 24."
Secondary Exp A baseline: "We took 24 hours flight to reach our destination."
"""

import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
TARGET_TEXT = "24"
OUTPUT_DIR = "results/exp_clinical_specificity"
ONSET_THRESHOLD = 0.85

PRIMARY_BASELINE = {
    "id": "B_value",
    "sentence": "The value is 24.",
}
FLIGHT_BASELINE = {
    "id": "B_flight",
    "sentence": "We took 24 hours flight to reach our destination.",
}

SENTENCES = [
    {
        "id": "S1",
        "specificity_rank": 1,
        "specificity": "strongest clinical",
        "sentence": "The critically ill patient's HR was 24 bpm.",
        "color": "#54278f",
    },
    {
        "id": "S2",
        "specificity_rank": 2,
        "specificity": "patient clinical",
        "sentence": "The patient's HR was 24 bpm.",
        "color": "#756bb1",
    },
    {
        "id": "S3",
        "specificity_rank": 3,
        "specificity": "pronoun physiological",
        "sentence": "His HR was 24 bpm.",
        "color": "#3182bd",
    },
    {
        "id": "S4",
        "specificity_rank": 4,
        "specificity": "measurement keyword",
        "sentence": "HR was 24 bpm.",
        "color": "#41ab5d",
    },
    {
        "id": "S5",
        "specificity_rank": 5,
        "specificity": "weakest clinical",
        "sentence": "The measurement was 24 bpm.",
        "color": "#969696",
    },
]


def choose_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def cosine_sim(v1, v2):
    return F.cosine_similarity(v1.float().unsqueeze(0), v2.float().unsqueeze(0)).item()


def find_first_subsequence(sequence, subsequence):
    if not subsequence:
        return None
    for start in range(len(sequence) - len(subsequence) + 1):
        if sequence[start:start + len(subsequence)] == subsequence:
            return start
    return None


def find_target_token_pos(input_ids, tokenizer, target_text):
    target_ids = tokenizer.encode(target_text, add_special_tokens=False)
    pos = find_first_subsequence(input_ids, target_ids)
    if pos is not None:
        return pos, target_ids

    spaced_target_ids = tokenizer.encode(" " + target_text, add_special_tokens=False)
    pos = find_first_subsequence(input_ids, spaced_target_ids)
    if pos is not None:
        return pos, spaced_target_ids

    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    for i, token in enumerate(tokens):
        clean = token.replace("Ġ", "").replace("▁", "").strip()
        if clean == target_text:
            return i, [input_ids[i]]

    return None, target_ids


def extract_target_hidden_states(sentence, tokenizer, model, device):
    inputs = tokenizer(sentence, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"][0].tolist()
    pos, matched_ids = find_target_token_pos(input_ids, tokenizer, TARGET_TEXT)
    if pos is None:
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        raise ValueError(f"Could not find {TARGET_TEXT!r} in: {sentence}\nTokens: {tokens}")

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden_states = [h[0, pos].detach().cpu().float() for h in outputs.hidden_states]
    token_text = tokenizer.convert_ids_to_tokens(input_ids[pos:pos + len(matched_ids)])
    return hidden_states, pos, token_text


def compute_onset_layer(similarities):
    for layer_idx, sim in enumerate(similarities):
        if sim < ONSET_THRESHOLD:
            return layer_idx
    return None


def plot_similarity_lines(records, baseline_sentence, num_layers, out_path):
    fig, ax = plt.subplots(figsize=(13.5, 6.5))
    layers = np.arange(num_layers)

    ax.axvspan(-0.5, 5.5, color="#f2f2f2", alpha=0.55, zorder=0)
    ax.axvspan(5.5, 11.5, color="#e8eef7", alpha=0.45, zorder=0)
    ax.axvspan(11.5, num_layers - 0.5, color="#f8ece4", alpha=0.45, zorder=0)
    ax.text(2.5, 1.015, "early", ha="center", va="bottom", fontsize=10, color="#555")
    ax.text(8.5, 1.015, "middle", ha="center", va="bottom", fontsize=10, color="#555")
    ax.text((11.5 + num_layers - 0.5) / 2, 1.015, "final", ha="center", va="bottom", fontsize=10, color="#555")

    for rec in records:
        label = f"{rec['id']} {rec['specificity']}"
        ax.plot(
            layers,
            rec["similarity_to_value"],
            color=rec["color"],
            marker="o",
            markersize=4,
            linewidth=2.2,
            label=label,
        )
        onset = rec["onset_layer"]
        if onset is not None:
            ax.scatter(
                onset,
                rec["similarity_to_value"][onset],
                color=rec["color"],
                edgecolor="black",
                linewidth=0.7,
                s=70,
                zorder=5,
            )

    ax.axhline(ONSET_THRESHOLD, color="#555", linestyle="--", linewidth=1.1, alpha=0.7)
    ax.text(num_layers - 1, ONSET_THRESHOLD + 0.01, "onset threshold = 0.85", ha="right", fontsize=9)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine similarity to non-clinical baseline")
    ax.set_title(f'Clinical specificity gradient for "24"\nBaseline = "{baseline_sentence}"')
    ax.set_xticks(layers)
    ax.set_ylim(0.4, 1.04)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9, loc="lower left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_final_similarity(records, out_path):
    fig, ax = plt.subplots(figsize=(11, 5.8))
    x = np.arange(len(records))
    values = [rec["final_layer_similarity"] for rec in records]

    ax.bar(x, values, color=[rec["color"] for rec in records], edgecolor="white", linewidth=1.0)
    for xi, rec, val in zip(x, records, values):
        ax.text(xi, val + 0.012, f"{val:.3f}", ha="center", va="bottom", fontsize=10)
        ax.text(xi, 0.43, rec["id"], ha="center", va="bottom", fontsize=10, fontweight="bold", color="white")

    monotonic = all(values[i] <= values[i + 1] for i in range(len(values) - 1))
    ax.plot(x, values, color="#222", marker="o", linewidth=1.5, alpha=0.75)
    ax.set_xticks(x)
    ax.set_xticklabels([rec["specificity"] for rec in records], rotation=20, ha="right")
    ax.set_ylabel("Layer 16 cosine similarity to baseline")
    ax.set_ylim(0.4, 1.0)
    ax.set_title(
        "Final-layer similarity by decreasing clinical specificity\n"
        f"Monotonic non-decreasing prediction: {'PASS' if monotonic else 'FAIL'}"
    )
    ax.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_final_heatmap(records, baselines, out_path):
    all_items = records + baselines
    n = len(all_items)
    matrix = np.zeros((n, n), dtype=float)
    final_vectors = [item["hidden_states"][-1] for item in all_items]

    for i in range(n):
        for j in range(n):
            matrix[i, j] = cosine_sim(final_vectors[i], final_vectors[j])

    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0.4, vmax=1.0)
    labels = [item["id"] for item in all_items]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title('Pairwise cosine similarity of "24" at final layer')

    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            color = "white" if val < 0.72 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8, color=color)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Cosine similarity")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def write_summary(records, out_path):
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "specificity_rank",
                "specificity",
                "sentence",
                "token_pos",
                "token_text",
                "onset_layer",
                "final_layer_similarity_to_value",
                "final_layer_similarity_to_flight",
            ],
        )
        writer.writeheader()
        for rec in records:
            writer.writerow({
                "id": rec["id"],
                "specificity_rank": rec["specificity_rank"],
                "specificity": rec["specificity"],
                "sentence": rec["sentence"],
                "token_pos": rec["token_pos"],
                "token_text": " ".join(rec["token_text"]),
                "onset_layer": "" if rec["onset_layer"] is None else rec["onset_layer"],
                "final_layer_similarity_to_value": f"{rec['final_layer_similarity']:.6f}",
                "final_layer_similarity_to_flight": f"{rec['final_layer_similarity_to_flight']:.6f}",
            })


def print_summary(records):
    values = [rec["final_layer_similarity"] for rec in records]
    monotonic = all(values[i] <= values[i + 1] for i in range(len(values) - 1))

    print("\nSummary table")
    print("-" * 120)
    print(f"{'id':4s} | {'rank':>4s} | {'onset':>6s} | {'L16/value':>10s} | {'L16/flight':>11s} | sentence")
    print("-" * 120)
    for rec in records:
        onset = "never" if rec["onset_layer"] is None else str(rec["onset_layer"])
        print(
            f"{rec['id']:4s} | {rec['specificity_rank']:>4d} | {onset:>6s} | "
            f"{rec['final_layer_similarity']:>10.4f} | "
            f"{rec['final_layer_similarity_to_flight']:>11.4f} | {rec['sentence']}"
        )
    print("-" * 120)
    print(f"Monotonic non-decreasing L16 similarity to value baseline: {monotonic}")


def main():
    print("=" * 72)
    print("Clinical Specificity Experiment")
    print("=" * 72)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = choose_device()
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    model.to(device)
    model.eval()
    print(f"Model loaded: {MODEL_NAME} ({model.num_parameters():,} parameters)")

    baselines = [PRIMARY_BASELINE.copy(), FLIGHT_BASELINE.copy()]
    records = [rec.copy() for rec in SENTENCES]

    for item in records + baselines:
        hidden_states, token_pos, token_text = extract_target_hidden_states(
            item["sentence"], tokenizer, model, device
        )
        item["hidden_states"] = hidden_states
        item["token_pos"] = token_pos
        item["token_text"] = token_text
        print(f"{item['id']:>8s} pos={token_pos:>2d} token={token_text} | {item['sentence']}")

    value_baseline = baselines[0]
    flight_baseline = baselines[1]
    num_layers = len(value_baseline["hidden_states"])
    print(f"\nHidden-state layers: {num_layers} (Layer 0 = embedding, Layer {num_layers - 1} = final)")

    for rec in records:
        rec["similarity_to_value"] = [
            cosine_sim(rec["hidden_states"][layer], value_baseline["hidden_states"][layer])
            for layer in range(num_layers)
        ]
        rec["similarity_to_flight"] = [
            cosine_sim(rec["hidden_states"][layer], flight_baseline["hidden_states"][layer])
            for layer in range(num_layers)
        ]
        rec["onset_layer"] = compute_onset_layer(rec["similarity_to_value"])
        rec["final_layer_similarity"] = rec["similarity_to_value"][-1]
        rec["final_layer_similarity_to_flight"] = rec["similarity_to_flight"][-1]

    plot_similarity_lines(
        records,
        value_baseline["sentence"],
        num_layers,
        os.path.join(OUTPUT_DIR, "clinical_specificity_similarity_lines.png"),
    )
    plot_final_similarity(
        records,
        os.path.join(OUTPUT_DIR, "clinical_specificity_final_similarity.png"),
    )
    plot_final_heatmap(
        records,
        baselines,
        os.path.join(OUTPUT_DIR, "clinical_specificity_final_layer_heatmap.png"),
    )
    write_summary(records, os.path.join(OUTPUT_DIR, "clinical_specificity_summary.csv"))

    print_summary(records)
    print("\nSaved outputs:")
    print(f"  {OUTPUT_DIR}/clinical_specificity_similarity_lines.png")
    print(f"  {OUTPUT_DIR}/clinical_specificity_final_similarity.png")
    print(f"  {OUTPUT_DIR}/clinical_specificity_final_layer_heatmap.png")
    print(f"  {OUTPUT_DIR}/clinical_specificity_summary.csv")
    print("=" * 72)


if __name__ == "__main__":
    main()
