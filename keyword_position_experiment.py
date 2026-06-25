"""
Keyword Position Experiment
===========================

Test whether the position of the domain keyword "heart rate" before vs. after
the number changes the hidden-state representation of the token "24".

Model: Llama 3.2-1B-Instruct
Baseline: Group C sentence 1, "The value is 24."
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
BASELINE_ID = "C1"
ONSET_THRESHOLD = 0.85
OUTPUT_DIR = "results/exp_keyword_position"


SENTENCE_GROUPS = {
    "A_keyword_before": [
        "Heart rate: 24 beats per minute.",
        "The heart rate was 24 beats per minute.",
        "Cardiac monitoring showed a heart rate of 24 beats per minute.",
        "The patient's heart rate measured 24 beats per minute.",
    ],
    "B_keyword_after": [
        "24 beats per minute was the recorded heart rate.",
        "24 was the patient's heart rate in beats per minute.",
        "The reading was 24, which represents the heart rate.",
        "Recorded value: 24, context is heart rate.",
    ],
    "C_no_keyword": [
        "The value is 24.",
        "24 was recorded.",
        "The reading is 24.",
        "The number is 24.",
    ],
}


GROUP_LABELS = {
    "A_keyword_before": "Group A: keyword before 24",
    "B_keyword_after": "Group B: keyword after 24",
    "C_no_keyword": "Group C: no keyword",
}


GROUP_COLORS = {
    "A_keyword_before": ["#08306b", "#2171b5", "#4292c6", "#9ecae1"],
    "B_keyword_after": ["#7f2704", "#d94801", "#f16913", "#fdae6b"],
    "C_no_keyword": ["#252525", "#636363", "#969696", "#bdbdbd"],
}


def choose_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def cosine_sim(v1, v2):
    return F.cosine_similarity(
        v1.float().unsqueeze(0),
        v2.float().unsqueeze(0),
    ).item()


def find_first_subsequence(sequence, subsequence):
    if not subsequence:
        return None
    limit = len(sequence) - len(subsequence) + 1
    for start in range(max(limit, 0)):
        if sequence[start:start + len(subsequence)] == subsequence:
            return start
    return None


def find_target_token_pos(input_ids, tokenizer, target_text):
    """
    Find the first token position of target_text in input_ids.

    If "24" tokenizes into multiple tokens, this returns the first token of the
    first matching token-id subsequence.
    """
    target_ids = tokenizer.encode(target_text, add_special_tokens=False)
    pos = find_first_subsequence(input_ids, target_ids)
    if pos is not None:
        return pos, target_ids

    # Fallback for tokenizers that include context-sensitive leading-space ids.
    target_ids_with_space = tokenizer.encode(" " + target_text, add_special_tokens=False)
    pos = find_first_subsequence(input_ids, target_ids_with_space)
    if pos is not None:
        return pos, target_ids_with_space

    # Last-resort textual fallback for debugging odd tokenization cases.
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    for i, tok in enumerate(tokens):
        clean = tok.replace("Ġ", "").replace("▁", "").strip()
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

    hidden_states = [layer_hs[0, pos].detach().cpu().float() for layer_hs in outputs.hidden_states]
    token_text = tokenizer.convert_ids_to_tokens(input_ids[pos:pos + len(matched_ids)])
    return hidden_states, pos, token_text


def make_records():
    records = []
    for group_key, sentences in SENTENCE_GROUPS.items():
        group_letter = group_key[0]
        for idx, sentence in enumerate(sentences, start=1):
            records.append({
                "id": f"{group_letter}{idx}",
                "group": group_key,
                "sentence": sentence,
                "color": GROUP_COLORS[group_key][idx - 1],
            })
    return records


def compute_onset_layer(similarities):
    for layer_idx, sim in enumerate(similarities):
        if sim < ONSET_THRESHOLD:
            return layer_idx
    return None


def plot_similarity_lines(records, num_layers, out_path):
    fig, ax = plt.subplots(figsize=(15, 7))
    layers = np.arange(num_layers)

    ax.axvspan(-0.5, 5.5, color="#f2f2f2", alpha=0.55, zorder=0)
    ax.axvspan(5.5, 11.5, color="#e8eef7", alpha=0.45, zorder=0)
    ax.axvspan(11.5, num_layers - 0.5, color="#f8ece4", alpha=0.45, zorder=0)
    ax.text(2.5, 1.015, "early", ha="center", va="bottom", fontsize=10, color="#555")
    ax.text(8.5, 1.015, "middle", ha="center", va="bottom", fontsize=10, color="#555")
    ax.text((11.5 + num_layers - 0.5) / 2, 1.015, "final", ha="center", va="bottom", fontsize=10, color="#555")

    for rec in records:
        label = f"{rec['id']} {rec['sentence']}"
        ax.plot(
            layers,
            rec["similarities"],
            color=rec["color"],
            linewidth=2,
            marker="o",
            markersize=4,
            label=label,
        )
        onset = rec["onset_layer"]
        if onset is not None:
            ax.scatter(
                onset,
                rec["similarities"][onset],
                color=rec["color"],
                edgecolor="black",
                linewidth=0.7,
                s=65,
                zorder=5,
            )

    ax.axhline(ONSET_THRESHOLD, color="#555", linestyle="--", linewidth=1.2, alpha=0.75)
    ax.text(num_layers - 1, ONSET_THRESHOLD + 0.01, "onset threshold = 0.85", ha="right", fontsize=9)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine similarity to baseline")
    ax.set_title(
        'Keyword Position Experiment: hidden state of "24"\n'
        'Baseline = C1 "The value is 24."'
    )
    ax.set_xticks(layers)
    ax.set_ylim(0.35, 1.04)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7.5, loc="lower left", ncol=1)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_onset_layers(records, out_path):
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ordered = sorted(records, key=lambda rec: (rec["group"], rec["id"]))
    y = np.arange(len(ordered))

    for yi, rec in zip(y, ordered):
        onset = rec["onset_layer"]
        x = onset if onset is not None else max(len(rec["similarities"]) - 1, 0) + 0.45
        marker = "o" if onset is not None else ">"
        ax.scatter(x, yi, color=rec["color"], marker=marker, s=95, edgecolor="black", linewidth=0.6)
        text = "never" if onset is None else str(onset)
        ax.text(x + 0.15, yi, f"onset {text}; final {rec['final_layer_similarity']:.3f}", va="center", fontsize=8)

    ax.axvline(ONSET_THRESHOLD, color="none")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{rec['id']}  {rec['sentence']}" for rec in ordered], fontsize=8)
    ax.set_xlabel("First layer where similarity drops below 0.85")
    ax.set_title('Onset layer for contextual divergence of "24"')
    ax.set_xlim(-0.5, len(ordered[0]["similarities"]) + 1)
    ax.set_xticks(range(len(ordered[0]["similarities"])))
    ax.grid(True, axis="x", alpha=0.25)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_final_layer_heatmap(records, out_path):
    n = len(records)
    matrix = np.zeros((n, n), dtype=float)
    final_vectors = [rec["hidden_states"][-1] for rec in records]

    for i in range(n):
        for j in range(n):
            matrix[i, j] = cosine_sim(final_vectors[i], final_vectors[j])

    fig, ax = plt.subplots(figsize=(11.5, 9.5))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0.35, vmax=1.0)
    labels = [rec["id"] for rec in records]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title('Pairwise cosine similarity of "24" at final layer')

    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            color = "white" if val < 0.72 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=7, color=color)

    # Group boundaries: after A4 and B4.
    for boundary in [3.5, 7.5]:
        ax.axhline(boundary, color="black", linewidth=1.1)
        ax.axvline(boundary, color="black", linewidth=1.1)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Cosine similarity")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def write_summary_csv(records, out_path):
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "group", "sentence", "token_pos", "token_text", "onset_layer", "final_layer_similarity"],
        )
        writer.writeheader()
        for rec in records:
            writer.writerow({
                "id": rec["id"],
                "group": rec["group"],
                "sentence": rec["sentence"],
                "token_pos": rec["token_pos"],
                "token_text": " ".join(rec["token_text"]),
                "onset_layer": "" if rec["onset_layer"] is None else rec["onset_layer"],
                "final_layer_similarity": f"{rec['final_layer_similarity']:.6f}",
            })


def print_summary(records):
    print("\nSummary table")
    print("-" * 110)
    print(f"{'id':4s} | {'group':18s} | {'onset':>6s} | {'final':>8s} | sentence")
    print("-" * 110)
    for rec in records:
        onset = "never" if rec["onset_layer"] is None else str(rec["onset_layer"])
        print(
            f"{rec['id']:4s} | {rec['group']:18s} | {onset:>6s} | "
            f"{rec['final_layer_similarity']:>8.4f} | {rec['sentence']}"
        )


def print_group_pairwise(records):
    print("\nWithin-group final-layer pairwise similarity")
    print("-" * 70)
    for group_key in SENTENCE_GROUPS:
        group_records = [rec for rec in records if rec["group"] == group_key]
        vals = []
        for i in range(len(group_records)):
            for j in range(i + 1, len(group_records)):
                vals.append(cosine_sim(group_records[i]["hidden_states"][-1], group_records[j]["hidden_states"][-1]))
        mean_val = float(np.mean(vals)) if vals else np.nan
        min_val = float(np.min(vals)) if vals else np.nan
        max_val = float(np.max(vals)) if vals else np.nan
        print(f"{GROUP_LABELS[group_key]:34s} mean={mean_val:.4f}  min={min_val:.4f}  max={max_val:.4f}")


def main():
    print("=" * 72)
    print("Keyword Position Experiment")
    print("=" * 72)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = choose_device()
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    model.to(device)
    model.eval()
    print(f"Model loaded: {MODEL_NAME} ({model.num_parameters():,} parameters)")

    records = make_records()
    for rec in records:
        hidden_states, token_pos, token_text = extract_target_hidden_states(
            rec["sentence"], tokenizer, model, device
        )
        rec["hidden_states"] = hidden_states
        rec["token_pos"] = token_pos
        rec["token_text"] = token_text
        print(f"{rec['id']:>2s} pos={token_pos:>2d} token={token_text} | {rec['sentence']}")

    baseline = next(rec for rec in records if rec["id"] == BASELINE_ID)
    num_layers = len(baseline["hidden_states"])
    print(f"\nHidden-state layers: {num_layers} (Layer 0 = embedding, Layer {num_layers - 1} = final)")
    print(f"Baseline: {BASELINE_ID} {baseline['sentence']}")

    for rec in records:
        rec["similarities"] = [
            cosine_sim(rec["hidden_states"][layer_idx], baseline["hidden_states"][layer_idx])
            for layer_idx in range(num_layers)
        ]
        rec["onset_layer"] = compute_onset_layer(rec["similarities"])
        rec["final_layer_similarity"] = rec["similarities"][-1]

    plot_similarity_lines(
        records,
        num_layers,
        os.path.join(OUTPUT_DIR, "keyword_position_similarity_lines.png"),
    )
    plot_onset_layers(
        records,
        os.path.join(OUTPUT_DIR, "keyword_position_onset_layers.png"),
    )
    plot_final_layer_heatmap(
        records,
        os.path.join(OUTPUT_DIR, "keyword_position_final_layer_heatmap.png"),
    )
    write_summary_csv(records, os.path.join(OUTPUT_DIR, "keyword_position_summary.csv"))

    print_summary(records)
    print_group_pairwise(records)

    print("\nSaved outputs:")
    print(f"  {OUTPUT_DIR}/keyword_position_similarity_lines.png")
    print(f"  {OUTPUT_DIR}/keyword_position_onset_layers.png")
    print(f"  {OUTPUT_DIR}/keyword_position_final_layer_heatmap.png")
    print(f"  {OUTPUT_DIR}/keyword_position_summary.csv")
    print("=" * 72)


if __name__ == "__main__":
    main()
