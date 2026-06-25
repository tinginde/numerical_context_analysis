"""
Token Position Control Experiment
=================================

Compare neutral, full-term, abbreviation, and clinical framings while keeping
the target number "24" in a closely matched sentence position.

Baseline: "The value was 24."
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
OUTPUT_DIR = "results/exp_token_position_control"
BASELINE_ID = "Neutral"

SENTENCES = [
    {
        "id": "Neutral",
        "label": "neutral",
        "sentence": "The value was 24.",
        "color": "#636363",
    },
    {
        "id": "FullTerm",
        "label": "full term",
        "sentence": "The heart rate was 24.",
        "color": "#3182bd",
    },
    {
        "id": "Abbrev",
        "label": "abbreviation",
        "sentence": "The HR reading was 24.",
        "color": "#756bb1",
    },
    {
        "id": "Clinical",
        "label": "clinical",
        "sentence": "The critically ill HR was 24.",
        "color": "#de2d26",
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
    all_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    return hidden_states, pos, token_text, all_tokens


def plot_lines(records, num_layers, out_path):
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    layers = np.arange(num_layers)

    for rec in records:
        ax.plot(
            layers,
            rec["similarity_to_baseline"],
            marker="o",
            markersize=4,
            linewidth=2.2,
            color=rec["color"],
            label=f"{rec['id']}: {rec['sentence']}",
        )

    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine similarity to neutral baseline")
    ax.set_title('Token-position controlled framing effect on "24"\nBaseline = "The value was 24."')
    ax.set_xticks(layers)
    ax.set_ylim(0.45, 1.04)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9, loc="lower left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_final_bar(records, out_path):
    fig, ax = plt.subplots(figsize=(8.5, 5.3))
    x = np.arange(len(records))
    vals = [rec["final_layer_similarity"] for rec in records]

    ax.bar(x, vals, color=[rec["color"] for rec in records], edgecolor="white", linewidth=1.0)
    for xi, val in zip(x, vals):
        ax.text(xi, val + 0.012, f"{val:.3f}", ha="center", va="bottom", fontsize=11)

    ax.set_xticks(x)
    ax.set_xticklabels([rec["label"] for rec in records])
    ax.set_ylabel("Layer 16 cosine similarity to neutral baseline")
    ax.set_ylim(0.45, 1.04)
    ax.set_title('Final-layer similarity of "24" with matched target position')
    ax.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def write_summary(records, out_path):
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "label",
                "sentence",
                "token_pos",
                "token_text",
                "tokens",
                "final_layer_similarity",
            ],
        )
        writer.writeheader()
        for rec in records:
            writer.writerow({
                "id": rec["id"],
                "label": rec["label"],
                "sentence": rec["sentence"],
                "token_pos": rec["token_pos"],
                "token_text": " ".join(rec["token_text"]),
                "tokens": " ".join(rec["tokens"]),
                "final_layer_similarity": f"{rec['final_layer_similarity']:.6f}",
            })


def main():
    print("=" * 72)
    print("Token Position Control Experiment")
    print("=" * 72)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = choose_device()
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    model.to(device)
    model.eval()
    print(f"Model loaded: {MODEL_NAME} ({model.num_parameters():,} parameters)")

    records = [rec.copy() for rec in SENTENCES]
    for rec in records:
        hidden_states, token_pos, token_text, tokens = extract_target_hidden_states(
            rec["sentence"], tokenizer, model, device
        )
        rec["hidden_states"] = hidden_states
        rec["token_pos"] = token_pos
        rec["token_text"] = token_text
        rec["tokens"] = tokens
        print(f"{rec['id']:>8s} pos={token_pos:>2d} token={token_text} | {rec['sentence']}")
        print(f"         tokens: {tokens}")

    baseline = next(rec for rec in records if rec["id"] == BASELINE_ID)
    num_layers = len(baseline["hidden_states"])
    for rec in records:
        rec["similarity_to_baseline"] = [
            cosine_sim(rec["hidden_states"][layer], baseline["hidden_states"][layer])
            for layer in range(num_layers)
        ]
        rec["final_layer_similarity"] = rec["similarity_to_baseline"][-1]

    plot_lines(records, num_layers, os.path.join(OUTPUT_DIR, "token_position_similarity_lines.png"))
    plot_final_bar(records, os.path.join(OUTPUT_DIR, "token_position_final_similarity.png"))
    write_summary(records, os.path.join(OUTPUT_DIR, "token_position_summary.csv"))

    print("\nSummary table")
    print("-" * 86)
    print(f"{'id':10s} | {'label':14s} | {'pos':>3s} | {'L16':>7s} | sentence")
    print("-" * 86)
    for rec in records:
        print(
            f"{rec['id']:10s} | {rec['label']:14s} | {rec['token_pos']:>3d} | "
            f"{rec['final_layer_similarity']:>7.4f} | {rec['sentence']}"
        )

    print("\nSaved outputs:")
    print(f"  {OUTPUT_DIR}/token_position_similarity_lines.png")
    print(f"  {OUTPUT_DIR}/token_position_final_similarity.png")
    print(f"  {OUTPUT_DIR}/token_position_summary.csv")
    print("=" * 72)


if __name__ == "__main__":
    main()
