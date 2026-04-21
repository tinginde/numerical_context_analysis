"""
Contextual Numerical Representation Analysis v2
=============================================
Adds Experiment C: the same context, but numbers with very different
clinical or physical meanings.

Core question: can the LLM distinguish numbers with different danger
levels within the same context?

Five medical contexts:
    1. Blood Pressure (mmHg)           40 / 79 / 120 / 180
    2. Body Temperature (°C)           25 / 35 / 37 / 41
    3. Heart Rate (bpm)                30 / 55 / 75 / 180
    4. Blood Glucose (mg/dL)           40 / 90 / 180 / 400
    5. Respiratory Rate (breaths/min)  4 / 16 / 25 / 40

Usage: place this file in your llm-internals-tutorial folder and run:
    python numerical_context_analysis_v2.py
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch.nn.functional as F
from matplotlib.lines import Line2D
from transformers import AutoTokenizer, AutoModelForCausalLM

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 70)
print("Contextual Numerical Representation Analysis v2")
print("Experiment C: same context, numbers with different clinical meanings")
print("=" * 70)

# ============================================================
# 1. Load the model
# ============================================================
print("\n[Step 1] Loading model...")

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("  Using CUDA GPU")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("  Using Apple MPS")
else:
    device = torch.device("cpu")
    print("  Using CPU")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
)
model.to(device)
model.eval()
print(f"  Model loaded ({model.num_parameters():,} parameters)")

# ============================================================
# 2. Define five medical contexts
# ============================================================

EXPERIMENT_C = [
    {
        "name": "Blood Pressure (mmHg)",
        "short": "BP",
        "template": "The patient's blood pressure was {} mmHg.",
        "values":   ["40",  "79",         "120",    "180"],
        "clinical": ["Crisis Low", "Hypotension", "Normal", "Hypertensive Crisis"],
        "normal_idx": 2,   # 120 is the baseline
    },
    {
        "name": "Body Temperature (°C)",
        "short": "Temp",
        "template": "The patient's body temperature was {} degrees Celsius.",
        "values":   ["25",              "35",             "37",     "41"],
        "clinical": ["Severe Hypothermia", "Mild Hypothermia", "Normal", "High Fever"],
        "normal_idx": 2,   # 37 is the baseline
    },
    {
        "name": "Heart Rate (bpm)",
        "short": "HR",
        "template": "The patient's heart rate was {} beats per minute.",
        "values":   ["30",               "55",        "75",     "180"],
        "clinical": ["Severe Bradycardia", "Low Normal", "Normal", "Severe Tachycardia"],
        "normal_idx": 2,   # 75 is the baseline
    },
    {
        "name": "Blood Glucose (mg/dL)",
        "short": "Glucose",
        "template": "The patient's blood glucose level was {} milligrams per deciliter.",
        "values":   ["40",                "90",     "180",          "400"],
        "clinical": ["Hypoglycemic Crisis", "Normal", "Hyperglycemia", "DKA Risk"],
        "normal_idx": 1,   # 90 is the baseline
    },
    {
        "name": "Respiratory Rate (breaths/min)",
        "short": "RR",
        "template": "The patient's respiratory rate was {} breaths per minute.",
        "values":   ["4",          "16",     "25",         "40"],
        "clinical": ["Near Apnea", "Normal", "Tachypnea", "Respiratory Crisis"],
        "normal_idx": 1,   # 16 is the baseline
    },
]

# ============================================================
# 3. Helper functions
# ============================================================

def get_hidden_states(sentence):
    inputs = tokenizer(sentence, return_tensors="pt").to(device)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].tolist())
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hs = [h[0].cpu() for h in outputs.hidden_states]
    return hs, tokens


def find_token_pos(tokens, target):
    """Find the target number token position, ignoring Ġ / ▁ prefixes."""
    for i, t in enumerate(tokens):
        clean = t.replace("Ġ", "").replace("▁", "").strip()
        if clean == target:
            return i
    return None


def cosine_sim(v1, v2):
    return F.cosine_similarity(
        v1.float().unsqueeze(0),
        v2.float().unsqueeze(0)
    ).item()


def run_group(group):
    """Run all numbers for one context and return each number's vector at every layer."""
    all_hs = []
    num_layers_found = None

    for val in group["values"]:
        sent = group["template"].format(val)
        hs, tokens = get_hidden_states(sent)

        pos = find_token_pos(tokens, val)
        if pos is None:
            pos = find_token_pos(tokens, val[:2])
        if pos is None:
            print(f"  ⚠ Could not find '{val}', tokens={tokens}")
            all_hs.append(None)
            continue

        all_hs.append([hs[l][pos] for l in range(len(hs))])
        if num_layers_found is None:
            num_layers_found = len(hs)

    return all_hs, num_layers_found

# ============================================================
# 4. Run all experiments
# ============================================================
print("\n[Step 2] Running Experiment C...")

results_C = []
num_layers = None

for group in EXPERIMENT_C:
    print(f"\n  {group['name']}")
    print(f"  {'Value':8s} | {'Clinical Meaning':25s} | Sentence")
    print("  " + "-" * 65)
    for val, clin in zip(group["values"], group["clinical"]):
        sent = group["template"].format(val)
        print(f"  {val:8s} | {clin:25s} | {sent[:45]}...")

    all_hs, nl = run_group(group)
    results_C.append(all_hs)
    if num_layers is None and nl is not None:
        num_layers = nl

print(f"\n  Total {num_layers} layers (Layer 0 = embedding, Layer {num_layers-1} = final layer)")

# ============================================================
# 5. Plot the results
# ============================================================
print("\n[Step 3] Generating plots...")

layers = list(range(num_layers))

# Colors indicate semantic position relative to each context's normal value.
SEMANTIC_STYLES = {
    "far_low":  {"color": '#1a5276', "label": "Far Low"},
    "low_side": {"color": '#7fb3d3', "label": "Low-side"},
    "normal":   {"color": '#27ae60', "label": "Normal (=1.0)"},
    "high_side": {"color": '#f39c12', "label": "High-side"},
    "far_high": {"color": '#e74c3c', "label": "Far High"},
}
MARKERS = ['v', 's', 'o', '^']


def get_value_semantic(group, value_idx):
    normal_idx = group["normal_idx"]
    if value_idx == normal_idx:
        return "normal"
    if value_idx < normal_idx:
        return "far_low" if value_idx == 0 else "low_side"
    return "far_high" if value_idx == len(group["values"]) - 1 else "high_side"


def get_value_color(group, value_idx):
    return SEMANTIC_STYLES[get_value_semantic(group, value_idx)]["color"]


def get_relative_position(group, value_idx):
    return value_idx - group["normal_idx"]

fig = plt.figure(figsize=(22, 20))
outer_gs = gridspec.GridSpec(3, 1, figure=fig, hspace=0.5)

# --- Top half: line charts for the five contexts (2 rows) ---
top_gs = gridspec.GridSpecFromSubplotSpec(
    2, 3, subplot_spec=outer_gs[0:2], hspace=0.55, wspace=0.38
)

for g_idx, (group, all_hs) in enumerate(zip(EXPERIMENT_C, results_C)):
    row, col = divmod(g_idx, 3)
    ax = fig.add_subplot(top_gs[row, col])

    normal_idx = group["normal_idx"]
    normal_val = group["values"][normal_idx]

    if all_hs[normal_idx] is None:
        ax.set_title(f"{group['short']}: data error")
        continue

    for v_idx, (val, clin, hs_vecs) in enumerate(
            zip(group["values"], group["clinical"], all_hs)):
        if hs_vecs is None:
            continue
        sims = [cosine_sim(all_hs[normal_idx][l], hs_vecs[l])
                for l in range(num_layers)]
        color = get_value_color(group, v_idx)
        lw = 2.5 if v_idx == normal_idx else 1.8
        ls = '--' if v_idx == normal_idx else '-'
        ax.plot(layers, sims,
                marker=MARKERS[v_idx % len(MARKERS)],
                markersize=4,
                linewidth=lw,
                linestyle=ls,
                label=f"{val} ({clin})",
                color=color)

    ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.4, linewidth=1)
    ax.set_title(f"{group['name']}\nBaseline = {normal_val} (Normal)", fontsize=10)
    ax.set_xlabel("Layer", fontsize=9)
    ax.set_ylabel("Cosine Similarity", fontsize=9)
    ax.set_xticks(layers[::2])
    ax.legend(fontsize=7.5, loc='lower left')
    ax.grid(True, alpha=0.25)
    ax.set_ylim(0.25, 1.08)

# Use the sixth panel for explanatory notes
ax_note = fig.add_subplot(top_gs[1, 2])
ax_note.axis('off')
ax_note.text(0.05, 0.95,
    "How to read:\n\n"
    "• Each line = one numerical value\n"
    "• Baseline (dashed) = context-specific normal value\n"
    "• Y-axis = cosine similarity to normal\n\n"
    "Color rule:\n"
    "• Blue tones = below normal\n"
    "• Green = normal\n"
    "• Orange/red = above normal\n\n"
    "Key question:\n"
    "Do values farther from each context's\n"
    "normal baseline have LOWER similarity\n"
    "than values closer to normal?\n\n"
    "If yes → LLM's representation\n"
    "is sensitive to clinical severity!",
    transform=ax_note.transAxes,
    fontsize=9.5,
    va='top',
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
)

# --- Bottom half: summary bar chart + monotonicity scatter ---
bottom_gs = gridspec.GridSpecFromSubplotSpec(
    1, 2, subplot_spec=outer_gs[2], wspace=0.4
)

# ---- Summary Bar Chart ----
ax_bar = fig.add_subplot(bottom_gs[0])

group_names = [g["short"] for g in EXPERIMENT_C]
n_groups = len(EXPERIMENT_C)
bar_width = 0.18
x = np.arange(n_groups)

legend_order = ["far_low", "low_side", "normal", "high_side", "far_high"]

for g_idx, (group, all_hs) in enumerate(zip(EXPERIMENT_C, results_C)):
    normal_idx = group["normal_idx"]
    if all_hs[normal_idx] is None:
        continue

    for v_idx, (val, hs_vecs) in enumerate(zip(group["values"], all_hs)):
        if hs_vecs is None:
            continue
        sim = cosine_sim(all_hs[normal_idx][-1], hs_vecs[-1])
        x_pos = g_idx + get_relative_position(group, v_idx) * bar_width
        color = get_value_color(group, v_idx)

        ax_bar.bar(x_pos, sim,
                   width=bar_width,
                   color=color,
                   alpha=0.88,
                   edgecolor='white',
                   linewidth=0.5)
        ax_bar.text(x_pos, sim + 0.004, f"{sim:.2f}",
                    ha='center', va='bottom', fontsize=6.5)

ax_bar.axhline(y=1.0, color='gray', linestyle='--', alpha=0.4)
ax_bar.set_xticks(range(n_groups))
ax_bar.set_xticklabels(group_names, fontsize=12)
ax_bar.set_ylabel("Cosine Similarity to Normal (Last Layer)", fontsize=10)
ax_bar.set_title(
    "Summary: How Far is Each Value from 'Normal' in LLM's Internal Space?\n"
    "(Last Layer — colors are assigned relative to each context-specific normal value)",
    fontsize=10
)
bar_handles = [
    Line2D([0], [0], marker='s', color='w',
           markerfacecolor=SEMANTIC_STYLES[key]["color"],
           markersize=9, label=SEMANTIC_STYLES[key]["label"])
    for key in legend_order
]
ax_bar.legend(handles=bar_handles, fontsize=9)
ax_bar.grid(True, alpha=0.25, axis='y')
ax_bar.set_ylim(0.25, 1.08)

# ---- U-shape Monotonicity ----
ax_mono = fig.add_subplot(bottom_gs[1])

scatter_colors = ['#1a5276', '#e74c3c', '#27ae60', '#f39c12', '#8e44ad']
scatter_markers = ['o', 's', '^', 'D', 'v']

for g_idx, (group, all_hs) in enumerate(zip(EXPERIMENT_C, results_C)):
    normal_idx = group["normal_idx"]
    if all_hs[normal_idx] is None:
        continue

    x_vals = []
    sims_last = []
    for v_idx, hs_vecs in enumerate(all_hs):
        if hs_vecs is None:
            continue
        x_vals.append(get_relative_position(group, v_idx))
        sims_last.append(cosine_sim(all_hs[normal_idx][-1], hs_vecs[-1]))

    ax_mono.plot(
        x_vals,
        sims_last,
        marker=scatter_markers[g_idx],
        markersize=8,
        linewidth=2,
        label=group["short"],
        color=scatter_colors[g_idx]
    )

ax_mono.axvline(x=0, color='gray', linestyle=':', alpha=0.4)
ax_mono.set_xlabel("Relative Position to Each Context's Normal Value\n(negative = below normal, 0 = normal, positive = above normal)", fontsize=9)
ax_mono.set_ylabel("Cosine Similarity to Normal (Last Layer)", fontsize=10)
ax_mono.set_title(
    "U-shape Test: Are Values Farther from Normal Least Similar?\n"
    "(Each context is re-centered so its own normal value sits at x = 0)",
    fontsize=10
)
ax_mono.legend(fontsize=9)
ax_mono.grid(True, alpha=0.3)
ax_mono.set_xticks([-2, -1, 0, 1, 2])
ax_mono.set_xticklabels(["Far\nLow", "Low-side", "Normal", "High-side", "Far\nHigh"])
ax_mono.set_ylim(0.25, 1.08)

plt.suptitle(
    "Experiment C: Same Medical Context, Clinically Different Numbers\n"
    "Does LLM Distinguish Dangerous vs Normal Values in Its Internal Representations?",
    fontsize=14, fontweight='bold'
)

os.makedirs("results/exp_C", exist_ok=True)
output_path = "results/exp_C/numerical_context_analysis_v2.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"  Plot saved to: {output_path}")
plt.close()

# ============================================================
# 6. Text summary
# ============================================================
print("\n" + "=" * 70)
print("Numerical Summary: Experiment C (final layer, normal value as baseline)")
print("=" * 70)

for group, all_hs in zip(EXPERIMENT_C, results_C):
    normal_idx = group["normal_idx"]
    normal_val = group["values"][normal_idx]
    if all_hs[normal_idx] is None:
        continue

    print(f"\n{group['name']}  (baseline = {normal_val})")
    print(f"  {'Value':8s} | {'Clinical Meaning':25s} | {'L0':>7} | {'L{}'.format(num_layers//2):>7} | {'L{}'.format(num_layers-1):>7}")
    print("  " + "-" * 62)

    for v_idx, (val, clin, hs_vecs) in enumerate(
            zip(group["values"], group["clinical"], all_hs)):
        if hs_vecs is None:
            print(f"  {val:8s} | {clin:25s} | {'N/A':>7}")
            continue
        l0  = cosine_sim(all_hs[normal_idx][0], hs_vecs[0])
        lM  = cosine_sim(all_hs[normal_idx][num_layers//2], hs_vecs[num_layers//2])
        lN  = cosine_sim(all_hs[normal_idx][-1], hs_vecs[-1])
        tag = " ← normal (baseline)" if v_idx == normal_idx else ""
        print(f"  {val:8s} | {clin:25s} | {l0:>7.4f} | {lM:>7.4f} | {lN:>7.4f}{tag}")

print("""
Three core questions:

  Q1. How low is the similarity between dangerous values and normal values?
      -> Lower values mean the LLM representation better distinguishes danger vs normal

  Q2. Does it show a U-shape?
      -> Normal-value similarity = 1.0
      -> The dangerous values at both ends should be lower than values closer to normal
      -> If so, the LLM representation space has clinical ordering

  Q3. Which medical context is the LLM most sensitive to, and which is it least sensitive to?
      -> Compare the last-layer similarity drop across the five contexts
      -> That becomes the section 4 findings for your paper
""")

print("=" * 70)
print("Done! Output image: numerical_context_analysis_v2.png")
print("=" * 70)
