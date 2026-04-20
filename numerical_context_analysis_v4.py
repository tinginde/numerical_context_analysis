"""
Contextual Numerical Representation Analysis v4
PCA Visualization + 1B vs 3B Model Comparison
Diabetes-Related Medical Values

8 Contexts × 6 Values = 48 Data Points per Model
"""

import gc
import os

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from transformers import AutoModelForCausalLM, AutoTokenizer

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ============================================================
# 1. Define 8 medical contexts (6 values each)
# ============================================================

CONTEXTS = [
    # --- Core diabetes-related biomarkers ---
    {
        "name": "Blood Glucose (mg/dL)",
        "short": "Glucose",
        "template": "The patient's blood glucose level was {} milligrams per deciliter.",
        "values": ["40", "70", "90", "180", "300", "500"],
        "clinical": [
            "Hypoglycemic Crisis",
            "Low Normal",
            "Normal",
            "Hyperglycemia",
            "Severe Hyperglycemia",
            "DKA Risk",
        ],
        "normal_idx": 2,
    },
    {
        "name": "HbA1c (%)",
        "short": "HbA1c",
        "template": "The patient's HbA1c level was {} percent.",
        "values": ["4.0", "5.0", "5.6", "6.5", "9.0", "14.0"],
        "clinical": [
            "Critically Low",
            "Low Normal",
            "Normal",
            "Pre-diabetes",
            "Diabetes",
            "Severe Diabetes",
        ],
        "normal_idx": 2,
    },
    {
        "name": "Fasting Insulin (uU/mL)",
        "short": "Insulin",
        "template": "The patient's fasting insulin level was {} micro units per milliliter.",
        "values": ["2", "5", "10", "25", "60", "120"],
        "clinical": [
            "Critically Low",
            "Low Normal",
            "Normal",
            "Insulin Resistance",
            "Severe Resistance",
            "Crisis",
        ],
        "normal_idx": 2,
    },
    {
        "name": "Triglycerides (mg/dL)",
        "short": "TG",
        "template": "The patient's triglyceride level was {} milligrams per deciliter.",
        "values": ["50", "100", "150", "250", "500", "1000"],
        "clinical": [
            "Very Low",
            "Normal Low",
            "Normal",
            "Borderline High",
            "High",
            "Very High",
        ],
        "normal_idx": 2,
    },
    # --- Common diabetes-related complications ---
    {
        "name": "Creatinine (mg/dL)",
        "short": "Creatinine",
        "template": "The patient's serum creatinine level was {} milligrams per deciliter.",
        "values": ["0.4", "0.7", "1.0", "1.5", "3.0", "8.0"],
        "clinical": [
            "Very Low",
            "Low Normal",
            "Normal",
            "Mild Elevation",
            "Renal Impairment",
            "Kidney Failure",
        ],
        "normal_idx": 2,
    },
    {
        "name": "Blood Pressure (mmHg)",
        "short": "BP",
        "template": "The patient's blood pressure was {} millimeters of mercury.",
        "values": ["60", "90", "120", "140", "180", "220"],
        "clinical": ["Shock", "Low Normal", "Normal", "Elevated", "Hypertension", "Crisis"],
        "normal_idx": 2,
    },
    # --- General vital signs ---
    {
        "name": "Heart Rate (bpm)",
        "short": "HR",
        "template": "The patient's heart rate was {} beats per minute.",
        "values": ["30", "50", "72", "100", "140", "180"],
        "clinical": [
            "Severe Bradycardia",
            "Low Normal",
            "Normal",
            "Elevated",
            "Tachycardia",
            "Severe Tachycardia",
        ],
        "normal_idx": 2,
    },
    {
        "name": "Body Temperature (C)",
        "short": "Temp",
        "template": "The patient's body temperature was {} degrees Celsius.",
        "values": ["34", "36", "37", "38", "39", "41"],
        "clinical": [
            "Hypothermia",
            "Low Normal",
            "Normal",
            "Mild Fever",
            "Fever",
            "High Fever",
        ],
        "normal_idx": 2,
    },
]

# Colors: danger index 0-5 maps to critical-low -> normal -> critical-high
DANGER_COLORS = ["#1a5276", "#3498db", "#27ae60", "#f39c12", "#e67e22", "#c0392b"]
DANGER_LABELS = [
    "0 Critical Low",
    "1 Low Normal",
    "2 Normal",
    "3 Mild High",
    "4 High",
    "5 Critical High",
]

# Marker shapes: one distinct shape for each of the 8 contexts
CONTEXT_MARKERS = ["o", "s", "^", "D", "v", "P", "*", "X"]

MODELS = [
    ("meta-llama/Llama-3.2-1B-Instruct", "1B"),
    ("meta-llama/Llama-3.2-3B-Instruct", "3B"),
]


# ============================================================
# 2. Helper functions
# ============================================================


def find_value_token_pos(tokens, value_str):
    """
    Find the value token position, supporting both integers and decimals.
    Prefer an exact match; if none is found, fall back to the integer part.
    """
    # Exact match, ignoring Ġ / ▁ prefixes
    for i, t in enumerate(tokens):
        clean = t.replace("Ġ", "").replace("▁", "").strip()
        if clean == value_str:
            return i
    # Decimal fallback: first search for the integer part
    if "." in value_str:
        int_part = value_str.split(".")[0]
        for i, t in enumerate(tokens):
            clean = t.replace("Ġ", "").replace("▁", "").strip()
            if clean == int_part:
                return i
    return None


def extract_all_hidden_states(contexts, tokenizer, model, device):
        """
        Run all values across all contexts and return:
            data[c_idx][v_idx] = list of tensors (one tensor per layer, shape [hidden_dim])
            num_layers: int (including the embedding layer)
        """
    data = [[None] * len(ctx["values"]) for ctx in contexts]
    num_layers = None
    total = sum(len(ctx["values"]) for ctx in contexts)
    done = 0

    for c_idx, ctx in enumerate(contexts):
        for v_idx, val in enumerate(ctx["values"]):
            done += 1
            sent = ctx["template"].format(val)
            inputs = tokenizer(sent, return_tensors="pt").to(device)
            tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].tolist())

            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)

            hs = [h[0].cpu().float() for h in outputs.hidden_states]
            if num_layers is None:
                num_layers = len(hs)

            pos = find_value_token_pos(tokens, val)
            if pos is None:
                print(f"  ⚠  [{done}/{total}] Could not find token for '{val}', skipping. tokens={tokens[:12]}")
                continue

            data[c_idx][v_idx] = [hs[l][pos] for l in range(num_layers)]

        print(f"  [{c_idx+1}/8] {ctx['short']} complete")

    return data, num_layers


def collect_layer_vectors(data, num_layers):
        """
        Reshape data[c_idx][v_idx] into:
            vectors_by_layer[l] = np.array of shape (N_valid, hidden_dim)
            meta = list of (c_idx, v_idx) corresponding to the rows in vectors_by_layer
        """
    meta = []
    idx_map = []  # (c_idx, v_idx) in order

    for c_idx, row in enumerate(data):
        for v_idx, entry in enumerate(row):
            if entry is not None:
                idx_map.append((c_idx, v_idx))

    vectors_by_layer = []
    for l in range(num_layers):
        vecs = []
        for c_idx, v_idx in idx_map:
            vecs.append(data[c_idx][v_idx][l].numpy())
        vectors_by_layer.append(np.array(vecs))

    return vectors_by_layer, idx_map


# ============================================================
# 3. PCA scatter subplot
# ============================================================


def draw_pca_scatter(ax, vectors_np, meta, layer_label):
    """
    vectors_np: (N, D) array
    meta: list of (c_idx, v_idx)
    Returns total variance explained (PC1 + PC2)
    """
    pca = PCA(n_components=2)
    coords = pca.fit_transform(vectors_np)
    var = pca.explained_variance_ratio_

    for (c_idx, v_idx), coord in zip(meta, coords):
        ax.scatter(
            coord[0],
            coord[1],
            c=DANGER_COLORS[v_idx],
            marker=CONTEXT_MARKERS[c_idx],
            s=95,
            alpha=0.88,
            edgecolors="white",
            linewidths=0.5,
        )

    total_var = (var[0] + var[1]) * 100
    ax.set_title(
        f"{layer_label}\nPC1+PC2: {total_var:.1f}% variance", fontsize=9
    )
    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}%)", fontsize=8)
    ax.set_ylabel(f"PC2 ({var[1]*100:.1f}%)", fontsize=8)
    ax.grid(True, alpha=0.2)
    ax.tick_params(labelsize=7)
    return total_var


# ============================================================
# 4. Main program: load the two models in sequence
# ============================================================

print("=" * 65)
print("Contextual Numerical Representation Analysis v4  —  PCA + 1B vs 3B")
print("8 contexts × 6 values = 48 data points")
print("=" * 65)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA GPU\n")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple MPS\n")
else:
    device = torch.device("cpu")
    print("Using CPU\n")

all_model_results = {}  # label -> {"data": ..., "num_layers": ..., "vectors_by_layer": ..., "meta": ...}

for model_id, label in MODELS:
    print(f"[{label}] Loading {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
    model.to(device)
    model.eval()
    print(f"  Parameters: {model.num_parameters():,}")

    data, num_layers = extract_all_hidden_states(CONTEXTS, tokenizer, model, device)

    # Release GPU memory
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()

    vectors_by_layer, meta = collect_layer_vectors(data, num_layers)
    all_model_results[label] = {
        "num_layers": num_layers,
        "vectors_by_layer": vectors_by_layer,
        "meta": meta,
    }
    print(f"  Done, {num_layers} layers, valid points {len(meta)}/48\n")

# ============================================================
# 5. Plotting
#    Layout:
#      Row 0: 1B — PCA @ Layer 0 / mid / last
#      Row 1: 3B — PCA @ Layer 0 / mid / last
#      Row 2: variance-across-layers line plot + legend
# ============================================================
print("Generating plots...")

fig = plt.figure(figsize=(20, 18))
outer_gs = gridspec.GridSpec(
    3, 1, figure=fig, hspace=0.55, height_ratios=[1, 1, 0.6]
)

variance_curves = {}  # label -> list of total_var per layer (for bottom plot)

for row_i, label in enumerate(["1B", "3B"]):
    info = all_model_results[label]
    num_layers = info["num_layers"]
    vectors_by_layer = info["vectors_by_layer"]
    meta = info["meta"]

    # Three representative layers: embedding / middle / final
    l_early = 0
    l_mid = num_layers // 2
    l_late = num_layers - 1
    chosen = [
        (l_early, f"Layer {l_early}  (Embedding)"),
        (l_mid,   f"Layer {l_mid}  (Middle)"),
        (l_late,  f"Layer {l_late}  (Final)"),
    ]

    inner_gs = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=outer_gs[row_i], wspace=0.32
    )

    for col_i, (l_idx, l_label) in enumerate(chosen):
        ax = fig.add_subplot(inner_gs[0, col_i])
        draw_pca_scatter(ax, vectors_by_layer[l_idx], meta, l_label)
        # Add the model label to the ylabel in the leftmost column
        if col_i == 0:
            ax.set_ylabel(f"【{label} Model — {num_layers} layers】\nPC2", fontsize=9, fontweight="bold")

    # Compute variance for all layers for the bottom line plot
    var_all = []
    for l in range(num_layers):
        pca_tmp = PCA(n_components=2)
        pca_tmp.fit(vectors_by_layer[l])
        var_all.append(sum(pca_tmp.explained_variance_ratio_) * 100)
    variance_curves[label] = var_all

# ---- Bottom: variance line plot + legend ----
bottom_gs = gridspec.GridSpecFromSubplotSpec(
    1, 2, subplot_spec=outer_gs[2], wspace=0.38
)

# Left: variance explained across layers (1B vs 3B)
ax_var = fig.add_subplot(bottom_gs[0])
model_colors = {"1B": "#2980b9", "3B": "#e74c3c"}

for label, var_all in variance_curves.items():
    n = len(var_all)
    x = np.linspace(0, 100, n)  # Normalize depth to 0-100% for alignment
    ax_var.plot(
        x,
        var_all,
        label=f"Llama {label}",
        color=model_colors[label],
        linewidth=2,
        marker="o",
        markersize=3,
        alpha=0.85,
    )

ax_var.set_xlabel("Layer Depth (%)", fontsize=9)
ax_var.set_ylabel("PC1+PC2 Variance Explained (%)", fontsize=9)
ax_var.set_title(
    "Representational Separability Across Layers\n"
    "(Higher = 48 vectors more spread out → more context-sensitive)",
    fontsize=9,
)
ax_var.legend(fontsize=9)
ax_var.grid(True, alpha=0.3)
ax_var.set_xlim(0, 100)

# Right: legend (color = danger level, shape = context)
ax_leg = fig.add_subplot(bottom_gs[1])
ax_leg.axis("off")

danger_handles = [
    Line2D(
        [0], [0],
        marker="o", color="w",
        markerfacecolor=c, markersize=11,
        label=l,
    )
    for c, l in zip(DANGER_COLORS, DANGER_LABELS)
]
ctx_handles = [
    Line2D(
        [0], [0],
        marker=m, color="#555",
        markersize=9, linestyle="None",
        label=f"{ctx['short']}",
    )
    for m, ctx in zip(CONTEXT_MARKERS, CONTEXTS)
]

leg1 = ax_leg.legend(
    handles=danger_handles,
    title="Danger Level  (point color)",
    loc="upper left",
    fontsize=8.5,
    title_fontsize=9,
    ncol=2,
)
ax_leg.add_artist(leg1)
ax_leg.legend(
    handles=ctx_handles,
    title="Medical Context  (point shape)",
    loc="lower left",
    fontsize=8.5,
    title_fontsize=9,
    ncol=2,
)

plt.suptitle(
    "Numerical Context Analysis v4  —  PCA Visualization\n"
    "Llama 1B vs 3B  |  8 Diabetes-Related Medical Contexts × 6 Severity Values = 48 Points",
    fontsize=13,
    fontweight="bold",
)

os.makedirs("results", exist_ok=True)
out_path = "results/numerical_context_analysis_v4.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Plot saved to: {out_path}")
plt.close()

print("\nDone!")
