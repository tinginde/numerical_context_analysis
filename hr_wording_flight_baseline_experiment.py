"""
HR Wording Experiment with Flight Baseline
==========================================

Compare whether "HR", "heart rate", and "The heart rate" framings change the
hidden-state representation of "24" relative to the Exp A flight baseline.
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
OUTPUT_DIR = "results/exp_hr_wording_flight_baseline"
BASELINE = "We took 24 hours flight to reach our destination."

SENTENCES = [
    {
        "id": "HR",
        "label": "abbreviation",
        "sentence": "HR was 24 bpm.",
        "color": "#756bb1",
    },
    {
        "id": "HeartRate",
        "label": "full term",
        "sentence": "Heart rate was 24 bpm.",
        "color": "#3182bd",
    },
    {
        "id": "TheHeartRate",
        "label": "full term + article",
        "sentence": "The heart rate was 24.",
        "color": "#31a354",
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


def plot_lines(records, num_layers, out_path):
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    layers = np.arange(num_layers)

    for rec in records:
        ax.plot(
            layers,
            rec["similarity_to_flight"],
            marker="o",
            markersize=4,
            linewidth=2.2,
            color=rec["color"],
            label=f"{rec['id']}: {rec['sentence']}",
        )

    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine similarity to flight baseline")
    ax.set_title('HR wording effect on hidden state of "24"\nBaseline = flight sentence')
    ax.set_xticks(layers)
    ax.set_ylim(0.35, 1.04)
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
    ax.set_ylabel("Layer 16 cosine similarity to flight baseline")
    ax.set_ylim(0.35, 0.9)
    ax.set_title('Final-layer similarity of "24" by HR wording')
    ax.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def write_summary(records, out_path):
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "label", "sentence", "token_pos", "token_text", "final_layer_similarity"],
        )
        writer.writeheader()
        for rec in records:
            writer.writerow({
                "id": rec["id"],
                "label": rec["label"],
                "sentence": rec["sentence"],
                "token_pos": rec["token_pos"],
                "token_text": " ".join(rec["token_text"]),
                "final_layer_similarity": f"{rec['final_layer_similarity']:.6f}",
            })


def main():
    print("=" * 72)
    print("HR Wording Experiment with Flight Baseline")
    print("=" * 72)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = choose_device()
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    model.to(device)
    model.eval()
    print(f"Model loaded: {MODEL_NAME} ({model.num_parameters():,} parameters)")

    baseline_hs, baseline_pos, baseline_token = extract_target_hidden_states(BASELINE, tokenizer, model, device)
    print(f"Baseline pos={baseline_pos:>2d} token={baseline_token} | {BASELINE}")

    records = [rec.copy() for rec in SENTENCES]
    for rec in records:
        hidden_states, token_pos, token_text = extract_target_hidden_states(
            rec["sentence"], tokenizer, model, device
        )
        rec["hidden_states"] = hidden_states
        rec["token_pos"] = token_pos
        rec["token_text"] = token_text
        rec["similarity_to_flight"] = [
            cosine_sim(hidden_states[layer], baseline_hs[layer])
            for layer in range(len(baseline_hs))
        ]
        rec["final_layer_similarity"] = rec["similarity_to_flight"][-1]
        print(
            f"{rec['id']:>12s} pos={token_pos:>2d} token={token_text} "
            f"L16={rec['final_layer_similarity']:.4f} | {rec['sentence']}"
        )

    plot_lines(records, len(baseline_hs), os.path.join(OUTPUT_DIR, "hr_wording_similarity_lines.png"))
    plot_final_bar(records, os.path.join(OUTPUT_DIR, "hr_wording_final_similarity.png"))
    write_summary(records, os.path.join(OUTPUT_DIR, "hr_wording_summary.csv"))

    print("\nSaved outputs:")
    print(f"  {OUTPUT_DIR}/hr_wording_similarity_lines.png")
    print(f"  {OUTPUT_DIR}/hr_wording_final_similarity.png")
    print(f"  {OUTPUT_DIR}/hr_wording_summary.csv")
    print("=" * 72)


if __name__ == "__main__":
    main()
