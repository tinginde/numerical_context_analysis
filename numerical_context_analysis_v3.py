"""
數值語境表示分析 v3 (Llama-3.2-3B)
Contextual Numerical Representation Analysis
=============================================
實驗 C：相同語境，但數字代表截然不同的臨床/物理意義（使用 3B 模型）
核心問題：LLM 能不能區分「同一語境下，具有不同危險程度的數字」？

五個醫療語境：
  1. 血壓 (mmHg)       40 / 79 / 120 / 180
  2. 體溫 (°C)         25 / 35 / 37 / 41
  3. 心跳 (bpm)        30 / 55 / 75 / 180
  4. 血糖 (mg/dL)      40 / 90 / 180 / 400
  5. 呼吸頻率 (次/分)   4 / 16 / 25 / 40

使用方式：放入 llm-internals-tutorial 資料夾執行
  python numerical_context_analysis_v3.py
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 70)
print("數值語境表示分析 v3 (Llama-3.2-3B)")
print("實驗 C：相同語境，臨床意義不同的數字（3B 模型）")
print("=" * 70)

# ============================================================
# 1. 載入模型
# ============================================================
print("\n[步驟 1] 載入模型...")

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("  使用 CUDA GPU")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("  使用 Apple MPS")
else:
    device = torch.device("cpu")
    print("  使用 CPU")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
)
model.to(device)
model.eval()
print(f"  模型載入完成（{model.num_parameters():,} 參數）")

# ============================================================
# 2. 定義五個醫療語境
# ============================================================

EXPERIMENT_C = [
    {
        "name": "Blood Pressure (mmHg)",
        "short": "BP",
        "template": "The patient's blood pressure was {} mmHg.",
        "values":   ["40",  "79",         "120",    "180"],
        "clinical": ["Crisis Low", "Hypotension", "Normal", "Hypertensive Crisis"],
        "normal_idx": 2,   # 120 為基準
    },
    {
        "name": "Body Temperature (°C)",
        "short": "Temp",
        "template": "The patient's body temperature was {} degrees Celsius.",
        "values":   ["25",              "35",             "37",     "41"],
        "clinical": ["Severe Hypothermia", "Mild Hypothermia", "Normal", "High Fever"],
        "normal_idx": 2,   # 37 為基準
    },
    {
        "name": "Heart Rate (bpm)",
        "short": "HR",
        "template": "The patient's heart rate was {} beats per minute.",
        "values":   ["30",               "55",        "75",     "180"],
        "clinical": ["Severe Bradycardia", "Low Normal", "Normal", "Severe Tachycardia"],
        "normal_idx": 2,   # 75 為基準
    },
    {
        "name": "Blood Glucose (mg/dL)",
        "short": "Glucose",
        "template": "The patient's blood glucose level was {} milligrams per deciliter.",
        "values":   ["40",                "90",     "180",          "400"],
        "clinical": ["Hypoglycemic Crisis", "Normal", "Hyperglycemia", "DKA Risk"],
        "normal_idx": 1,   # 90 為基準
    },
    {
        "name": "Respiratory Rate (breaths/min)",
        "short": "RR",
        "template": "The patient's respiratory rate was {} breaths per minute.",
        "values":   ["4",          "16",     "25",         "40"],
        "clinical": ["Near Apnea", "Normal", "Tachypnea", "Respiratory Crisis"],
        "normal_idx": 1,   # 16 為基準
    },
]

# ============================================================
# 3. 工具函數
# ============================================================

def get_hidden_states(sentence):
    inputs = tokenizer(sentence, return_tensors="pt").to(device)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].tolist())
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hs = [h[0].cpu() for h in outputs.hidden_states]
    return hs, tokens


def find_token_pos(tokens, target):
    """找目標數字 token 的位置（忽略前綴 Ġ / ▁）"""
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
    """跑一個語境的所有數字，回傳每個數字在所有層的向量"""
    all_hs = []
    num_layers_found = None

    for val in group["values"]:
        sent = group["template"].format(val)
        hs, tokens = get_hidden_states(sent)

        pos = find_token_pos(tokens, val)
        if pos is None:
            pos = find_token_pos(tokens, val[:2])
        if pos is None:
            print(f"  ⚠ 找不到 '{val}'，tokens={tokens}")
            all_hs.append(None)
            continue

        all_hs.append([hs[l][pos] for l in range(len(hs))])
        if num_layers_found is None:
            num_layers_found = len(hs)

    return all_hs, num_layers_found

# ============================================================
# 4. 執行所有實驗
# ============================================================
print("\n[步驟 2] 執行實驗 C (3B model)...")

results_C = []
num_layers = None

for group in EXPERIMENT_C:
    print(f"\n  {group['name']}")
    print(f"  {'數值':8s} | {'臨床意義':25s} | 句子")
    print("  " + "-" * 65)
    for val, clin in zip(group["values"], group["clinical"]):
        sent = group["template"].format(val)
        print(f"  {val:8s} | {clin:25s} | {sent[:45]}...")

    all_hs, nl = run_group(group)
    results_C.append(all_hs)
    if num_layers is None and nl is not None:
        num_layers = nl

print(f"\n  共 {num_layers} 層（Layer 0 = embedding，Layer {num_layers-1} = 最後一層）")

# ============================================================
# 5. 畫圖
# ============================================================
print("\n[步驟 3] 繪製圖表...")

layers = list(range(num_layers))

# 顏色代表「偏離正常的程度」
# 最危險 → 深色，接近正常 → 淺色，正常 → 綠
COLOR_MAP = {
    0: '#1a5276',   # 危險低（深藍）
    1: '#7fb3d3',   # 偏低（淺藍）
    2: '#27ae60',   # 正常（綠）
    3: '#e74c3c',   # 危險高（紅）
}
MARKERS = ['v', 's', 'o', '^']

fig = plt.figure(figsize=(22, 20))
outer_gs = gridspec.GridSpec(3, 1, figure=fig, hspace=0.5)

# --- 上半部：五個語境的折線圖（2 列） ---
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
        color = COLOR_MAP.get(v_idx, '#7f8c8d')
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

# 第六格放說明文字
ax_note = fig.add_subplot(top_gs[1, 2])
ax_note.axis('off')
ax_note.text(0.05, 0.95,
    "How to read:\n\n"
    "• Each line = one numerical value\n"
    "• Baseline (dashed) = Normal value\n"
    "• Y-axis = cosine similarity to normal\n\n"
    "Key question:\n"
    "Do 'danger' values (dark blue / red)\n"
    "have LOWER similarity to 'normal'\n"
    "than 'mild' values (light colors)?\n\n"
    "If yes → LLM's representation\n"
    "is sensitive to clinical severity!",
    transform=ax_note.transAxes,
    fontsize=9.5,
    va='top',
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
)

# --- 下半部：Summary bar chart + monotonicity scatter ---
bottom_gs = gridspec.GridSpecFromSubplotSpec(
    1, 2, subplot_spec=outer_gs[2], wspace=0.4
)

# ---- Summary Bar Chart ----
ax_bar = fig.add_subplot(bottom_gs[0])

group_names = [g["short"] for g in EXPERIMENT_C]
n_groups = len(EXPERIMENT_C)
bar_width = 0.18
x = np.arange(n_groups)

bar_colors = list(COLOR_MAP.values())
bar_legend_labels = ["Danger Low", "Low/Mild", "Normal (=1.0)", "High/Danger"]

for v_idx in range(4):
    bar_vals = []
    x_pos = []
    for g_idx, (group, all_hs) in enumerate(zip(EXPERIMENT_C, results_C)):
        if v_idx >= len(group["values"]):
            continue
        normal_idx = group["normal_idx"]
        if all_hs[normal_idx] is None or all_hs[v_idx] is None:
            continue
        sim = cosine_sim(all_hs[normal_idx][-1], all_hs[v_idx][-1])
        bar_vals.append(sim)
        x_pos.append(g_idx + (v_idx - 1.5) * bar_width)

    if bar_vals:
        ax_bar.bar(x_pos, bar_vals,
                   width=bar_width,
                   color=bar_colors[v_idx],
                   alpha=0.88,
                   label=bar_legend_labels[v_idx],
                   edgecolor='white',
                   linewidth=0.5)
        for xp, yv in zip(x_pos, bar_vals):
            ax_bar.text(xp, yv + 0.004, f"{yv:.2f}",
                        ha='center', va='bottom', fontsize=6.5)

ax_bar.axhline(y=1.0, color='gray', linestyle='--', alpha=0.4)
ax_bar.set_xticks(range(n_groups))
ax_bar.set_xticklabels(group_names, fontsize=12)
ax_bar.set_ylabel("Cosine Similarity to Normal (Last Layer)", fontsize=10)
ax_bar.set_title(
    "Summary: How Far is Each Value from 'Normal' in LLM's Internal Space?\n"
    "(Last Layer — Lower bar = LLM treats value as more different from normal)",
    fontsize=10
)
ax_bar.legend(fontsize=9)
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

    sims_last = []
    for hs_vecs in all_hs:
        if hs_vecs is None:
            sims_last.append(np.nan)
        else:
            sims_last.append(cosine_sim(all_hs[normal_idx][-1], hs_vecs[-1]))

    ax_mono.plot(
        range(len(group["values"])),
        sims_last,
        marker=scatter_markers[g_idx],
        markersize=8,
        linewidth=2,
        label=group["short"],
        color=scatter_colors[g_idx]
    )

ax_mono.axvline(x=1.5, color='gray', linestyle=':', alpha=0.4)
ax_mono.set_xlabel("Value Index\n(0=danger low → 1=mild low → 2=normal → 3=danger high)", fontsize=9)
ax_mono.set_ylabel("Cosine Similarity to Normal (Last Layer)", fontsize=10)
ax_mono.set_title(
    "U-shape Test: Are Extreme Values Least Similar to Normal?\n"
    "(Expected: index 0 and 3 should be lower than index 1 and 2)",
    fontsize=10
)
ax_mono.legend(fontsize=9)
ax_mono.grid(True, alpha=0.3)
ax_mono.set_xticks([0, 1, 2, 3])
ax_mono.set_xticklabels(["Danger\nLow", "Mild\nLow", "Normal", "Danger\nHigh"])
ax_mono.set_ylim(0.25, 1.08)

plt.suptitle(
    "Experiment C (Llama-3.2-3B): Same Medical Context, Clinically Different Numbers\n"
    "Does LLM Distinguish Dangerous vs Normal Values in Its Internal Representations?",
    fontsize=14, fontweight='bold'
)

output_path = "numerical_context_analysis_v3.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"  圖表儲存為：{output_path}")
plt.close()

# ============================================================
# 6. 文字摘要
# ============================================================
print("\n" + "=" * 70)
print("數值摘要：實驗 C（最後一層，以正常值為基準）")
print("=" * 70)

for group, all_hs in zip(EXPERIMENT_C, results_C):
    normal_idx = group["normal_idx"]
    normal_val = group["values"][normal_idx]
    if all_hs[normal_idx] is None:
        continue

    print(f"\n{group['name']}  (baseline = {normal_val})")
    print(f"  {'數值':8s} | {'臨床意義':25s} | {'L0':>7} | {'L{}'.format(num_layers//2):>7} | {'L{}'.format(num_layers-1):>7}")
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
三個核心問題：

  Q1. 危險值 vs 正常值的 similarity 有多低？
      → 越低代表 LLM 的表示越能區分「危險」vs「正常」

  Q2. 是否呈現 U 型？
      → 正常值 similarity = 1.0
      → 兩端的危險值 similarity 應該都低於靠近正常的值
      → 如果是 → LLM 的表示空間有「臨床有序性」

  Q3. 哪個醫療語境 LLM 最敏感？哪個最不敏感？
      → 比較五個語境最後一層的 similarity 下降幅度
      → 這就是你 paper 的 section 4 findings 🎯
""")

print("=" * 70)
print("完成！產出圖片：numerical_context_analysis_v3.png")
print("=" * 70)
