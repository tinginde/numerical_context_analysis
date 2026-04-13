"""
數值語境表示分析 v5 — 行為層評估
Behavioral Evaluation: Accuracy / Consistency / Risk Alignment

設計矩陣：
  3 輸出格式 (numeric / category / hybrid)
× 4 模型角色 (zero-shot / role-prompting / instruction-constrained / forced-answer)
× 2 模型大小 (1B / 3B)
× 48 pairs  (8 語境 × 6 數值)
= 1152 次推論

修改說明：
  [Mod 1] category 改用 biomarker-specific 臨床標籤（非通用量表）
           → ground truth 更準確，測試真實臨床知識
  [Mod 2] 新增 forced_answer role：明確告知研究情境、非臨床診療
           → 隔離 1B 拒絕行為，讓結果可比
"""

import gc
import os
import re
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import f1_score
from transformers import AutoModelForCausalLM, AutoTokenizer

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ============================================================
# 1. 語境定義（8 個語境 × 6 數值）
#    clinical 欄位 = [Mod 1] biomarker-specific 臨床標籤
#                   index 0 = most critical low
#                   index 5 = most critical high
# ============================================================

CONTEXTS = [
    {
        "name": "Blood Glucose",
        "unit": "mg/dL",
        "short": "Glucose",
        "template": "The patient's blood glucose level was {value} milligrams per deciliter.",
        "values":   ["40",               "70",         "90",     "180",          "300",                   "500"],
        "clinical": ["Hypoglycemic Crisis", "Low Normal", "Normal", "Hyperglycemia", "Severe Hyperglycemia", "DKA Risk"],
    },
    {
        "name": "HbA1c",
        "unit": "%",
        "short": "HbA1c",
        "template": "The patient's HbA1c level was {value} percent.",
        "values":   ["4.0",           "5.0",        "5.6",    "6.5",         "9.0",      "14.0"],
        "clinical": ["Critically Low", "Low Normal", "Normal", "Pre-diabetes", "Diabetes", "Severe Diabetes"],
    },
    {
        "name": "Fasting Insulin",
        "unit": "uU/mL",
        "short": "Insulin",
        "template": "The patient's fasting insulin level was {value} micro units per milliliter.",
        "values":   ["2",              "5",          "10",     "25",                "60",                "120"],
        "clinical": ["Critically Low", "Low Normal", "Normal", "Insulin Resistance", "Severe Resistance", "Crisis"],
    },
    {
        "name": "Triglycerides",
        "unit": "mg/dL",
        "short": "TG",
        "template": "The patient's triglyceride level was {value} milligrams per deciliter.",
        "values":   ["50",       "100",        "150",    "250",             "500",   "1000"],
        "clinical": ["Very Low", "Normal Low", "Normal", "Borderline High", "High",  "Very High"],
    },
    {
        "name": "Serum Creatinine",
        "unit": "mg/dL",
        "short": "Creatinine",
        "template": "The patient's serum creatinine level was {value} milligrams per deciliter.",
        "values":   ["0.4",      "0.7",        "1.0",    "1.5",             "3.0",              "8.0"],
        "clinical": ["Very Low", "Low Normal", "Normal", "Mild Elevation",  "Renal Impairment", "Kidney Failure"],
    },
    {
        "name": "Blood Pressure",
        "unit": "mmHg",
        "short": "BP",
        "template": "The patient's blood pressure was {value} millimeters of mercury.",
        "values":   ["60",    "90",        "120",    "140",      "180",         "220"],
        "clinical": ["Shock", "Low Normal", "Normal", "Elevated", "Hypertension", "Crisis"],
    },
    {
        "name": "Heart Rate",
        "unit": "bpm",
        "short": "HR",
        "template": "The patient's heart rate was {value} beats per minute.",
        "values":   ["30",                 "50",        "72",     "100",      "140",         "180"],
        "clinical": ["Severe Bradycardia", "Low Normal", "Normal", "Elevated", "Tachycardia", "Severe Tachycardia"],
    },
    {
        "name": "Body Temperature",
        "unit": "°C",
        "short": "Temp",
        "template": "The patient's body temperature was {value} degrees Celsius.",
        "values":   ["34",          "36",        "37",     "38",         "39",    "41"],
        "clinical": ["Hypothermia", "Low Normal", "Normal", "Mild Fever", "Fever", "High Fever"],
    },
]

# ============================================================
# 2. 實驗條件
# ============================================================

# --- 輸出格式 ---
# [Mod 1] category 的 {labels} 由 build_prompt 動態填入 biomarker-specific 標籤
TASK_TEMPLATES = {
    "numeric": (
        "Clinical assessment task:\n"
        "Patient's {name} = {value} {unit}.\n\n"
        "Rate the clinical danger level on a scale of 0 to 5:\n"
        "  0 = Critically Low\n"
        "  1 = Low Normal\n"
        "  2 = Normal\n"
        "  3 = Mild High\n"
        "  4 = High\n"
        "  5 = Critically High\n\n"
        "Reply with ONLY a single digit (0, 1, 2, 3, 4, or 5)."
    ),
    "category": (
        "Clinical assessment task:\n"
        "Patient's {name} = {value} {unit}.\n\n"
        "Classify the clinical status.\n"
        "Reply with ONLY one of these exact labels:\n"
        "{labels}"
        # {labels} is filled per-context: e.g. "Hypoglycemic Crisis / Low Normal / Normal / ..."
    ),
    "hybrid": (
        "Clinical assessment task:\n"
        "Patient's {name} = {value} {unit}.\n\n"
        "Assess the clinical danger level.\n"
        "Reply in EXACTLY this format: \"<digit> - <label>\"\n"
        "  Digit : 0–5  (0 = most critical low, 2 = normal, 5 = most critical high)\n"
        "  Label : choose from: {labels}\n\n"
        "Example reply: \"4 - High\""
    ),
}

# --- 模型角色（含 [Mod 2] forced_answer）---
SYSTEM_PROMPTS = {
    "zero_shot": None,

    "role_prompting": (
        "You are an experienced clinical physician specializing in "
        "metabolic and endocrine disorders. Provide precise clinical assessments."
    ),

    "instruction_constrained": (
        "You must base your clinical judgment strictly on the numerical value provided. "
        "Do not make assumptions or rely on contextual priors beyond the given number."
    ),

    # [Mod 2] forced_answer：隔離拒絕行為，尤其針對 1B numeric
    "forced_answer": (
        "CONTEXT: This is a controlled academic research experiment studying how language "
        "models process numerical medical information. This is NOT a clinical decision-support "
        "tool and the output will NOT be used for any real medical decisions.\n\n"
        "You MUST provide the requested response format. Do not refuse, add safety disclaimers, "
        "or say you cannot answer. If uncertain, provide your best estimate based solely on "
        "the numerical value given."
    ),
}

MODELS = [
    ("meta-llama/Llama-3.2-1B-Instruct", "1B"),
    ("meta-llama/Llama-3.2-3B-Instruct", "3B"),
]

# ============================================================
# 3. Prompt 建構
# ============================================================

def build_labels_str(ctx):
    """把 biomarker-specific 臨床標籤組成 ' / ' 分隔字串。"""
    return " / ".join(ctx["clinical"])


def build_prompt(ctx, val, fmt_key, role_key, tokenizer):
    labels_str = build_labels_str(ctx)
    task = TASK_TEMPLATES[fmt_key].format(
        name=ctx["name"], value=val, unit=ctx["unit"],
        labels=labels_str,
    )
    system = SYSTEM_PROMPTS[role_key]

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": task})

    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

# ============================================================
# 4. 生成
# ============================================================

def generate_response(model, tokenizer, prompt, device, max_new_tokens=40):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0][input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

# ============================================================
# 5. 解析
#    [Mod 1] parse_category_ctx：依語境用 biomarker-specific 標籤比對
# ============================================================

def parse_numeric(text):
    """Extract first digit 0-5. Returns int or None."""
    m = re.search(r'\b([0-5])\b', text)
    return int(m.group(1)) if m else None


def parse_category_ctx(text, ctx_labels):
    """
    [Mod 1] Fuzzy-match against biomarker-specific labels.
    Longest match first (avoids 'High' eating 'Critically High').
    Returns index (0-5) into ctx_labels, or None.
    """
    t = text.lower().strip().rstrip(".")
    for label in sorted(ctx_labels, key=len, reverse=True):
        if label.lower() in t:
            return ctx_labels.index(label)
    return None


def parse_hybrid(text, ctx_labels):
    """
    Parse 'X - Label' format using biomarker-specific labels.
    Returns (numeric: int|None, category: int|None, is_consistent: bool|None)
    """
    m = re.search(r'\b([0-5])\s*[-–]\s*([A-Za-z ]+)', text)
    if m:
        num = int(m.group(1))
        cat = parse_category_ctx(m.group(2).strip(), ctx_labels)
        consistent = (cat == num) if cat is not None else None
        return num, cat, consistent
    # Fallback: parse each part independently
    return parse_numeric(text), parse_category_ctx(text, ctx_labels), None


def parse_output(raw_text, fmt_key, ctx):
    """
    Returns dict: numeric, category, hybrid_consistent (np.nan when missing).
    ctx is required for [Mod 1] biomarker-specific label matching.
    """
    result = {
        "numeric":           np.nan,
        "category":          np.nan,
        "hybrid_consistent": np.nan,
    }
    labels = ctx["clinical"]

    if fmt_key == "numeric":
        v = parse_numeric(raw_text)
        result["numeric"] = v if v is not None else np.nan

    elif fmt_key == "category":
        v = parse_category_ctx(raw_text, labels)
        result["category"] = v if v is not None else np.nan

    else:  # hybrid
        num, cat, cons = parse_hybrid(raw_text, labels)
        result["numeric"]           = num  if num  is not None else np.nan
        result["category"]          = cat  if cat  is not None else np.nan
        result["hybrid_consistent"] = bool(cons) if cons is not None else np.nan

    return result

# ============================================================
# 6. 推論主迴圈
# ============================================================

def run_experiment(model, tokenizer, device, model_label):
    rows = []
    combos = list(product(TASK_TEMPLATES.keys(), SYSTEM_PROMPTS.keys()))
    total = len(CONTEXTS) * 6 * len(combos)
    done  = 0

    for fmt_key, role_key in combos:
        condition = f"{fmt_key}__{role_key}"

        for c_idx, ctx in enumerate(CONTEXTS):
            for v_idx, val in enumerate(ctx["values"]):
                done += 1
                prompt = build_prompt(ctx, val, fmt_key, role_key, tokenizer)
                raw    = generate_response(model, tokenizer, prompt, device)
                parsed = parse_output(raw, fmt_key, ctx)   # [Mod 1] pass ctx

                rows.append({
                    "model":        model_label,
                    "format":       fmt_key,
                    "role":         role_key,
                    "condition":    condition,
                    "context":      ctx["short"],
                    "value":        val,
                    "v_idx":        v_idx,        # 0 = most critical low … 5 = most critical high
                    "ground_truth": v_idx,        # same: ordinal danger index
                    "gt_label":     ctx["clinical"][v_idx],   # [Mod 1] biomarker-specific label
                    "raw_output":   raw,
                    **parsed,
                })

                if done % 48 == 0 or done == total:
                    print(f"    [{done:>4}/{total}] {fmt_key:8s} | {role_key:25s} | "
                          f"{ctx['short']:10s} = {val}")
    return rows

# ============================================================
# 7. 指標計算
# ============================================================

def pred_column(fmt_key):
    return "numeric" if fmt_key in ("numeric", "hybrid") else "category"


def compute_metrics(df):
    records = []

    for (model_lbl, fmt_key, role_key), grp in df.groupby(
        ["model", "format", "role"]
    ):
        condition = f"{fmt_key}__{role_key}"
        pcol  = pred_column(fmt_key)
        valid = grp.dropna(subset=[pcol]).copy()
        valid[pcol] = valid[pcol].astype(float)
        pred  = valid[pcol]
        gt    = valid["ground_truth"].astype(float)

        # --- Accuracy (exact match on ordinal index) ---
        accuracy = (pred.round() == gt).mean() if len(valid) > 0 else np.nan

        # --- Consistency: mean Spearman ρ across 8 contexts ---
        rhos = []
        for _, cg in valid.groupby("context"):
            cg = cg.sort_values("v_idx")
            if len(cg) >= 3:
                rho, _ = spearmanr(cg["v_idx"], cg[pcol])
                if not np.isnan(rho):
                    rhos.append(rho)
        spearman_rho = float(np.mean(rhos)) if rhos else np.nan

        # --- Risk Alignment: F1 for extreme values (v_idx 0 or 5) ---
        is_ext_gt   = ((gt         == 0) | (gt         == 5)).astype(int)
        is_ext_pred = ((pred.round() == 0) | (pred.round() == 5)).astype(int)
        if len(valid) >= 2 and is_ext_gt.sum() > 0:
            risk_f1 = f1_score(is_ext_gt, is_ext_pred, zero_division=0)
        else:
            risk_f1 = np.nan

        # --- Hybrid internal consistency rate ---
        hyb_cons = np.nan
        if fmt_key == "hybrid":
            hv = grp.dropna(subset=["hybrid_consistent"])
            hyb_cons = float(hv["hybrid_consistent"].mean()) if len(hv) > 0 else np.nan

        records.append({
            "model":              model_lbl,
            "format":             fmt_key,
            "role":               role_key,
            "condition":          condition,
            "accuracy":           accuracy,
            "spearman_rho":       spearman_rho,
            "risk_f1":            risk_f1,
            "hybrid_consistency": hyb_cons,
            "n_valid":            len(valid),
            "n_total":            len(grp),
            "parse_rate":         len(valid) / len(grp) if len(grp) > 0 else np.nan,
        })

    return pd.DataFrame(records)

# ============================================================
# 8. 繪圖
# ============================================================

# 12 條件：3 formats × 4 roles
CONDITIONS_ORDERED = [
    "numeric__zero_shot",
    "numeric__role_prompting",
    "numeric__instruction_constrained",
    "numeric__forced_answer",
    "category__zero_shot",
    "category__role_prompting",
    "category__instruction_constrained",
    "category__forced_answer",
    "hybrid__zero_shot",
    "hybrid__role_prompting",
    "hybrid__instruction_constrained",
    "hybrid__forced_answer",
]
COND_SHORT = [
    "Zero-shot", "Role", "Constrained", "Forced",
    "Zero-shot", "Role", "Constrained", "Forced",
    "Zero-shot", "Role", "Constrained", "Forced",
]
GROUP_NAMES  = ["Numeric", "Category\n(biomarker-specific)", "Hybrid"]
GROUP_RANGES = [(0, 3), (4, 7), (8, 11)]   # index spans in CONDITIONS_ORDERED
GROUP_COLORS = ["#d6eaf8", "#d5f5e3", "#fdebd0"]

MODEL_COLORS = {"1B": "#2980b9", "3B": "#e74c3c"}

# Highlight forced_answer bars with a border
FORCED_INDICES = {3, 7, 11}   # positions of forced_answer in CONDITIONS_ORDERED


def _bar_fig(metrics_df, metric_col, ylabel, title, out_path,
             ylim, hline=None, hline_label=None):
    fig, ax = plt.subplots(figsize=(17, 5.8))
    x  = np.arange(len(CONDITIONS_ORDERED))
    bw = 0.33

    # Background group coloring
    for (start, end), color, gname in zip(GROUP_RANGES, GROUP_COLORS, GROUP_NAMES):
        ax.axvspan(start - 0.5, end + 0.5, color=color, alpha=0.35, zorder=0)
        ax.text((start + end) / 2, ylim[1] * 0.97, gname,
                ha="center", fontsize=9.5, color="#444", fontstyle="italic")

    for i, model_lbl in enumerate(["1B", "3B"]):
        sub  = metrics_df[metrics_df["model"] == model_lbl]
        vals = []
        for cond in CONDITIONS_ORDERED:
            row = sub[sub["condition"] == cond]
            vals.append(row[metric_col].values[0] if len(row) > 0 else np.nan)

        offset = (i - 0.5) * bw
        for xi, (v, cond_i) in enumerate(zip(vals, range(len(CONDITIONS_ORDERED)))):
            is_forced = cond_i in FORCED_INDICES
            bar = ax.bar(
                xi + offset, v if not np.isnan(v) else 0,
                bw,
                color=MODEL_COLORS[model_lbl],
                alpha=0.88,
                edgecolor="#e74c3c" if is_forced and model_lbl == "1B" else "white",
                linewidth=2.0 if is_forced else 0.6,
                zorder=3,
                label=f"Llama {model_lbl}" if xi == 0 else "_nolegend_",
            )
            if not np.isnan(v):
                ax.text(xi + offset, v + (ylim[1] - ylim[0]) * 0.012,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=7.2)

    if hline is not None:
        ax.axhline(hline, color="gray", linestyle="--", linewidth=1.2,
                   alpha=0.7, label=hline_label or f"= {hline:.3f}", zorder=2)

    # Forced answer annotation arrow on numeric group
    ax.annotate(
        "[Mod 2]\nForced Answer",
        xy=(3, ylim[0] + (ylim[1] - ylim[0]) * 0.08),
        xytext=(3, ylim[0] + (ylim[1] - ylim[0]) * 0.22),
        ha="center", fontsize=7.5, color="#c0392b",
        arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.2),
    )

    ax.set_xticks(x)
    ax.set_xticklabels(COND_SHORT, fontsize=8.5)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    ax.set_ylim(*ylim)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.2, axis="y", zorder=1)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  儲存：{out_path}")


def plot_parse_rate(metrics_df, out_path):
    _bar_fig(
        metrics_df, "parse_rate",
        ylabel="Parse Success Rate",
        title="Instruction-Following Rate — Did the model reply in the requested format?\n"
              "(NaN outputs excluded from all other metrics  |  [Mod 2] Forced Answer border = red)",
        out_path=out_path,
        ylim=(0, 1.12),
    )


def plot_hybrid_consistency(metrics_df, out_path):
    hyb = metrics_df[metrics_df["format"] == "hybrid"].copy()
    if hyb.empty or hyb["hybrid_consistency"].isna().all():
        print("  (Hybrid consistency: no valid data, skipping)")
        return

    roles        = ["zero_shot", "role_prompting", "instruction_constrained", "forced_answer"]
    role_labels  = ["Zero-shot", "Role Prompting", "Instruction\nConstrained", "Forced\nAnswer"]
    x  = np.arange(len(roles))
    bw = 0.33

    fig, ax = plt.subplots(figsize=(9, 4.8))
    for i, model_lbl in enumerate(["1B", "3B"]):
        sub  = hyb[hyb["model"] == model_lbl]
        vals = []
        for role in roles:
            row = sub[sub["role"] == role]
            vals.append(row["hybrid_consistency"].values[0] if len(row) > 0 else np.nan)

        offset = (i - 0.5) * bw
        for xi, (v, role) in enumerate(zip(vals, roles)):
            is_forced = role == "forced_answer"
            ax.bar(xi + offset, v if not np.isnan(v) else 0, bw,
                   color=MODEL_COLORS[model_lbl], alpha=0.88,
                   edgecolor="#e74c3c" if is_forced and model_lbl == "1B" else "white",
                   linewidth=2.0 if is_forced else 0.6,
                   label=f"Llama {model_lbl}" if xi == 0 else "_nolegend_")
            if not np.isnan(v):
                ax.text(xi + offset, v + 0.01, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(role_labels, fontsize=10)
    ax.set_ylabel("Internal Consistency Rate\n(digit label agrees with text label)", fontsize=9)
    ax.set_title("Hybrid Format — Internal Consistency\n"
                 "Does the digit match the written label? (e.g. '4 - Hypertension' ✓ vs '4 - Normal' ✗)\n"
                 "[Mod 1] Labels are biomarker-specific",
                 fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25, axis="y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  儲存：{out_path}")


# ============================================================
# 9. Main
# ============================================================

print("=" * 65)
print("數值語境表示分析 v5 — 行為層評估（修改版）")
print("3 格式 × 4 角色 × 2 模型 × 48 pairs = 1152 次推論")
print("[Mod 1] Category: biomarker-specific 臨床標籤")
print("[Mod 2] 新增 forced_answer role（隔離拒絕行為）")
print("=" * 65)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("使用 CUDA GPU\n")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("使用 Apple MPS\n")
else:
    device = torch.device("cpu")
    print("使用 CPU\n")

os.makedirs("results", exist_ok=True)
RAW_CSV     = "results/v5_raw.csv"
METRICS_CSV = "results/v5_metrics.csv"

all_rows = []

for model_id, model_label in MODELS:
    print(f"\n{'='*55}")
    print(f"[{model_label}] {model_id}")
    print("=" * 55)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model     = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16
    )
    model.to(device)
    model.eval()
    print(f"  參數量：{model.num_parameters():,}\n")

    rows = run_experiment(model, tokenizer, device, model_label)
    all_rows.extend(rows)

    # 每個模型跑完即存（防 crash）
    pd.DataFrame(all_rows).to_csv(RAW_CSV, index=False, encoding="utf-8-sig")
    print(f"\n  ✓ 原始結果儲存：{RAW_CSV}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

# --- 計算指標 ---
df_raw     = pd.read_csv(RAW_CSV)
df_metrics = compute_metrics(df_raw)
df_metrics.to_csv(METRICS_CSV, index=False, encoding="utf-8-sig")
print(f"\n✓ 指標儲存：{METRICS_CSV}")
print(df_metrics[["model", "format", "role",
                   "accuracy", "spearman_rho", "risk_f1",
                   "parse_rate"]].to_string(index=False))

# --- 繪圖 ---
print("\n繪製圖表...")

_bar_fig(
    df_metrics, "accuracy",
    ylabel="Accuracy  (output index matches ground-truth danger level)",
    title="【Accuracy】行為輸出是否分類正確？\n"
          "Llama 1B vs 3B  |  [Mod 1] Category uses biomarker-specific labels  "
          "|  chance = 1/6 ≈ 0.17",
    out_path="results/v5_accuracy.png",
    ylim=(0, 1.15),
    hline=1 / 6,
    hline_label="Chance level (1/6 ≈ 0.17)",
)

_bar_fig(
    df_metrics, "spearman_rho",
    ylabel="Spearman ρ  (monotonicity within each context)",
    title="【Consistency】輸出隨數值變動是否單調一致？\n"
          "Llama 1B vs 3B  |  ρ = 1 → 完全單調  |  ρ = 0 → 隨機",
    out_path="results/v5_consistency.png",
    ylim=(-0.3, 1.15),
    hline=0,
    hline_label="ρ = 0 (random)",
)

_bar_fig(
    df_metrics, "risk_f1",
    ylabel="F1 Score  (critical-low / critical-high detection)",
    title="【Risk Alignment】極端危險值是否被正確識別？\n"
          "Llama 1B vs 3B  |  Binary F1 for v_idx ∈ {0, 5}",
    out_path="results/v5_risk_alignment.png",
    ylim=(0, 1.15),
)

plot_parse_rate(df_metrics, out_path="results/v5_parse_rate.png")
plot_hybrid_consistency(df_metrics, out_path="results/v5_hybrid_consistency.png")

print("\n" + "=" * 55)
print("完成！輸出檔案：")
for f in [RAW_CSV, METRICS_CSV,
          "results/v5_accuracy.png",
          "results/v5_consistency.png",
          "results/v5_risk_alignment.png",
          "results/v5_parse_rate.png",
          "results/v5_hybrid_consistency.png"]:
    print(f"  {f}")
print("=" * 55)
