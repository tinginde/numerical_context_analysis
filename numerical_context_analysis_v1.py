"""
數值語境表示分析
Contextual Numerical Representation Analysis
=============================================
研究問題：同一個數字（如 "24"）在不同語境下，
LLM 的內部表示（hidden states）有沒有差異？

使用方式：直接放入你的 llm-internals-tutorial 資料夾執行
  python numerical_context_analysis.py

需要已安裝：torch, transformers, matplotlib, numpy, scikit-learn
需要已登入：huggingface-cli login
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
print("數值語境表示分析 | Contextual Numerical Representation Analysis")
print("=" * 70)

# ============================================================
# 1. 載入模型（與李老師教學相同的模型）
# ============================================================
print("\n[步驟 1] 載入模型...")

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

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
    torch_dtype=torch.float32,
)
model.to(device)
model.eval()
print(f"  模型載入完成（{model.num_parameters():,} 參數）")

# ============================================================
# 2. 定義實驗句子
# ============================================================

# 實驗 A：同一個數字 "24"，放在不同語境裡
EXPERIMENT_A_SENTENCES = [
    "We took 24 hours flight to reach our destination.",
    "He took 24 hours to learn a programming language.",
    "We ate 24 apples for breakfast.",
    "The patient's heart rate was 24 beats per minute.",
    "She saved 24 dollars from her allowance.",
]
EXPERIMENT_A_LABELS = [
    "24 hrs (flight)",
    "24 hrs (learning)",
    "24 apples",
    "24 bpm (medical)",
    "24 dollars",
]

# 實驗 B：同一語境，換三個不同數字（你原本做的實驗）
EXPERIMENT_B_SENTENCES = [
    "We took 2.4 hours flight to reach our destination.",
    "We took 24 hours flight to reach our destination.",
    "We took 240 hours flight to reach our destination.",
]
EXPERIMENT_B_LABELS = ["2.4 hrs", "24 hrs", "240 hrs"]
EXPERIMENT_B_NUMBERS = ["2", "24", "240"]   # 對應 tokenizer 切割後的第一個 token

# ============================================================
# 3. 工具函數
# ============================================================

def get_hidden_states(sentence):
    """
    回傳：
      hidden_states : list[Tensor], 長度 = num_layers+1
                      每個 Tensor 形狀 = [seq_len, hidden_dim]
      tokens        : list[str]
    """
    inputs = tokenizer(sentence, return_tensors="pt").to(device)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].tolist())
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hs = [h[0].cpu() for h in outputs.hidden_states]
    return hs, tokens


def find_token_pos(tokens, target):
    """找目標數字 token 的位置（忽略 tokenizer 加的空格前綴 Ġ / ▁）"""
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

# ============================================================
# 4. 跑實驗 A
# ============================================================
print("\n[步驟 2] 實驗 A：同一數字 '24'，不同語境")
print("-" * 60)

all_hs_A = []
num_layers = None

for sent, label in zip(EXPERIMENT_A_SENTENCES, EXPERIMENT_A_LABELS):
    hs, tokens = get_hidden_states(sent)
    pos = find_token_pos(tokens, "24")

    if pos is None:
        print(f"  ⚠ 找不到 '24'：{sent}")
        print(f"    Tokens: {tokens}")
        all_hs_A.append(None)
        continue

    print(f"  ✓ pos={pos:2d} | {label:20s} | {sent[:48]}...")
    all_hs_A.append([hs[l][pos] for l in range(len(hs))])

    if num_layers is None:
        num_layers = len(hs)

print(f"\n  共 {num_layers} 層（Layer 0 = embedding，Layer {num_layers-1} = 最後一層）")

# 以第一個句子為基準，計算各語境 vs 基準在每層的 cosine similarity
baseline_idx_A = 0
baseline_label_A = EXPERIMENT_A_LABELS[baseline_idx_A]
cosine_A = {}  # {label: [sim_l0, sim_l1, ...]}

for i, (hs_vecs, label) in enumerate(zip(all_hs_A, EXPERIMENT_A_LABELS)):
    if i == baseline_idx_A or hs_vecs is None:
        continue
    sims = [cosine_sim(all_hs_A[baseline_idx_A][l], hs_vecs[l])
            for l in range(num_layers)]
    cosine_A[label] = sims

# ============================================================
# 5. 跑實驗 B
# ============================================================
print("\n[步驟 3] 實驗 B：同一語境，不同數字")
print("-" * 60)

all_hs_B = []

for num_str, num_tok, sent, label in zip(
        EXPERIMENT_B_NUMBERS,
        EXPERIMENT_B_NUMBERS,
        EXPERIMENT_B_SENTENCES,
        EXPERIMENT_B_LABELS):
    hs, tokens = get_hidden_states(sent)
    pos = find_token_pos(tokens, num_tok)

    if pos is None:
        print(f"  ⚠ 找不到 '{num_tok}'：{sent}")
        print(f"    Tokens: {tokens}")
        all_hs_B.append(None)
        continue

    print(f"  ✓ pos={pos:2d} | {label:12s} | {sent[:50]}...")
    all_hs_B.append([hs[l][pos] for l in range(num_layers)])

# 以 "24 hrs"（index 1）為基準
baseline_idx_B = 1
baseline_label_B = EXPERIMENT_B_LABELS[baseline_idx_B]
cosine_B = {}

for i, (hs_vecs, label) in enumerate(zip(all_hs_B, EXPERIMENT_B_LABELS)):
    if i == baseline_idx_B or hs_vecs is None:
        continue
    sims = [cosine_sim(all_hs_B[baseline_idx_B][l], hs_vecs[l])
            for l in range(num_layers)]
    cosine_B[label] = sims

# ============================================================
# 6. 畫圖
# ============================================================
print("\n[步驟 4] 繪製圖表...")

layers = list(range(num_layers))
colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
markers = ['o', 's', '^', 'D', 'v']

fig = plt.figure(figsize=(16, 13))
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.5, wspace=0.35)

# ---- 圖 1：實驗 A 折線圖 ----
ax1 = fig.add_subplot(gs[0, :])
for idx, (label, sims) in enumerate(cosine_A.items()):
    ax1.plot(layers, sims,
             marker=markers[idx % len(markers)],
             markersize=5,
             linewidth=2,
             label=f"vs {label}",
             color=colors[idx % len(colors)])

ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.4, linewidth=1)
ax1.set_xlabel("Layer", fontsize=12)
ax1.set_ylabel("Cosine Similarity", fontsize=12)
ax1.set_title(
    f'Exp A: "24" in Different Semantic Contexts\n'
    f'Baseline = "{baseline_label_A}"  |  Lower similarity = more context-sensitive',
    fontsize=12
)
ax1.set_xticks(layers)
ax1.legend(fontsize=9, loc='lower left', ncol=2)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0.4, 1.05)

# ---- 圖 2：實驗 B 折線圖 ----
ax2 = fig.add_subplot(gs[1, 0])
for idx, (label, sims) in enumerate(cosine_B.items()):
    ax2.plot(layers, sims,
             marker=markers[idx % len(markers)],
             markersize=5,
             linewidth=2,
             label=f"vs {label}",
             color=colors[idx % len(colors)])

ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.4, linewidth=1)
ax2.set_xlabel("Layer", fontsize=12)
ax2.set_ylabel("Cosine Similarity", fontsize=12)
ax2.set_title(
    f'Exp B: Same Sentence, Different Numbers\n'
    f'Baseline = "{baseline_label_B}"',
    fontsize=12
)
ax2.set_xticks(layers)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0.4, 1.05)

# ---- 圖 3：最後一層 pairwise similarity heatmap（實驗 A）----
ax3 = fig.add_subplot(gs[1, 1])
last_layer = num_layers - 1
valid_idx = [i for i, x in enumerate(all_hs_A) if x is not None]
n_valid = len(valid_idx)
sim_matrix = np.zeros((n_valid, n_valid))

for r, i in enumerate(valid_idx):
    for c, j in enumerate(valid_idx):
        sim_matrix[r][c] = cosine_sim(all_hs_A[i][last_layer],
                                       all_hs_A[j][last_layer])

valid_labels = [EXPERIMENT_A_LABELS[i] for i in valid_idx]
im = ax3.imshow(sim_matrix, cmap='RdYlGn', vmin=0.6, vmax=1.0)
ax3.set_xticks(range(n_valid))
ax3.set_yticks(range(n_valid))
ax3.set_xticklabels(valid_labels, rotation=40, ha='right', fontsize=8)
ax3.set_yticklabels(valid_labels, fontsize=8)
ax3.set_title(
    f'Pairwise Similarity of "24"\nat Last Layer (L{last_layer})',
    fontsize=12
)
plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)

for r in range(n_valid):
    for c in range(n_valid):
        val = sim_matrix[r][c]
        color = 'white' if val < 0.8 else 'black'
        ax3.text(c, r, f"{val:.3f}", ha='center', va='center',
                 fontsize=7, color=color)

plt.suptitle(
    'Contextual Numerical Representation Analysis\n'
    'Does the same number "24" have different internal representations in different contexts?',
    fontsize=13, fontweight='bold'
)

output_path = "numerical_context_analysis.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"  圖表儲存為：{output_path}")
plt.close()

# ============================================================
# 7. 文字摘要
# ============================================================
print("\n" + "=" * 70)
print("數值摘要")
print("=" * 70)

print(f"\n實驗 A（基準：{baseline_label_A}）")
print(f"  {'比較對象':25s} | {'L0':>7} | {'L8':>7} | {'L{}'.format(num_layers-1):>7}")
print("  " + "-" * 52)
for label, sims in cosine_A.items():
    mid = sims[8] if len(sims) > 8 else sims[len(sims)//2]
    print(f"  {label:25s} | {sims[0]:>7.4f} | {mid:>7.4f} | {sims[-1]:>7.4f}")

print(f"\n實驗 B（基準：{baseline_label_B}）")
print(f"  {'比較對象':15s} | {'L0':>7} | {'L8':>7} | {'L{}'.format(num_layers-1):>7}")
print("  " + "-" * 42)
for label, sims in cosine_B.items():
    mid = sims[8] if len(sims) > 8 else sims[len(sims)//2]
    print(f"  {label:15s} | {sims[0]:>7.4f} | {mid:>7.4f} | {sims[-1]:>7.4f}")

print("""
解讀方式：
  - Layer 0 的 similarity 應接近 1.0（數字 token 的 embedding 相同）
  - 若後面幾層 similarity 下降 → 語境正在改變數字的表示
  - 下降越多 → 模型對語境越敏感

  研究假設：
  "24 hrs (flight)" vs "24 bpm (medical)" 的 similarity
  應低於 "24 hrs (flight)" vs "24 hrs (learning)"
  → 跨語義類別的差異 > 同語義類別的差異
""")

print("=" * 70)
print("完成！產出圖片：numerical_context_analysis.png")
print("=" * 70)
