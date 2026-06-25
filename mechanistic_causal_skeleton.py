"""
Mechanistic causal tests for numerical information collapse.

Core observation:
    deeper layers -> context dominates -> number information disappears

Question:
    which operation causes the collapse?

Candidate hypotheses:
    H_attn: attention writes context-dominant information into the number token.
    H_mlp: MLP amplifies context/risk directions or compresses numeric directions.
    H_norm: normalization flattens useful magnitude differences.

This is a runnable skeleton, not a final benchmark. The intended workflow is:
    1. Define paired prompts that isolate number vs context.
    2. Build metrics for "number information" and "context information".
    3. Localize layers with logit lens / representation metrics.
    4. Patch attn/mlp/norm activations and compare causal effect sizes.

Usage:
    python mechanistic_causal_skeleton.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
OUT_DIR = Path("results/exp_F_mechanistic")


# ============================================================
# 1. Experimental design
# ============================================================


@dataclass(frozen=True)
class PromptCase:
    name: str
    prompt: str
    number: str
    context_label: str
    severity_label: str


CASES = [
    PromptCase(
        name="heart_rate_low",
        prompt="The patient's heart rate was 24 beats per minute.",
        number="24",
        context_label="medical_heart_rate",
        severity_label="critical_low",
    ),
    PromptCase(
        name="heart_rate_normal",
        prompt="The patient's heart rate was 72 beats per minute.",
        number="72",
        context_label="medical_heart_rate",
        severity_label="normal",
    ),
    PromptCase(
        name="flight_hours_24",
        prompt="The flight duration was 24 hours.",
        number="24",
        context_label="travel_duration",
        severity_label="nonmedical",
    ),
    PromptCase(
        name="apples_24",
        prompt="The basket contained 24 apples.",
        number="24",
        context_label="counting_objects",
        severity_label="nonmedical",
    ),
]


@dataclass(frozen=True)
class PatchSpec:
    """Patch one module output at one layer and token position."""

    layer: int
    module_name: str  # one of: layer_out, input_ln, attn, post_attn_ln, mlp
    token_pos: int


@dataclass
class ForwardCache:
    tokens: List[str]
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    module_outputs: Dict[Tuple[str, int], torch.Tensor]
    hidden_states: List[torch.Tensor]
    attentions: Optional[List[torch.Tensor]]
    logits: torch.Tensor


# ============================================================
# 2. Model loading and token utilities
# ============================================================


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(model_name: str = MODEL_NAME):
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
    ).to(device)
    model.eval()
    return tokenizer, model, device


def clean_token(token: str) -> str:
    # Handles common sentencepiece / BPE visible whitespace markers.
    return token.replace("Ġ", "").replace("▁", "").strip()


def find_number_token_span(tokens: List[str], number: str) -> List[int]:
    """
    Return token positions whose concatenated cleaned string equals number.

    This handles simple multi-token numbers such as "2.4" or "1,200".
    If tokenization is surprising, print tokens and adjust this function.
    """
    for start in range(len(tokens)):
        acc = ""
        span = []
        for end in range(start, len(tokens)):
            acc += clean_token(tokens[end])
            span.append(end)
            if acc == number:
                return span
            if not number.startswith(acc):
                break
    raise ValueError(f"Could not find number={number!r} in tokens={tokens}")


def label_token_ids(tokenizer, labels: Iterable[str]) -> List[int]:
    """
    Map labels to first-token ids for simple logit-lens probes.

    For a real paper experiment, replace this with:
      - a learned linear probe,
      - multi-token logprob scoring,
      - or a calibrated contrast set.
    """
    ids = []
    for label in labels:
        encoded = tokenizer(" " + label, add_special_tokens=False)["input_ids"]
        if not encoded:
            raise ValueError(f"Could not tokenize label={label!r}")
        ids.append(encoded[0])
    return ids


# ============================================================
# 3. Hooking and activation capture
# ============================================================


def module_for(model, layer: int, module_name: str):
    block = model.model.layers[layer]
    if module_name == "layer_out":
        return block
    if module_name == "input_ln":
        return block.input_layernorm
    if module_name == "attn":
        return block.self_attn
    if module_name == "post_attn_ln":
        return block.post_attention_layernorm
    if module_name == "mlp":
        return block.mlp
    raise ValueError(f"Unknown module_name={module_name}")


def tensor_from_module_output(output):
    if isinstance(output, tuple):
        return output[0]
    return output


def replace_module_output(output, patched_tensor: torch.Tensor):
    if isinstance(output, tuple):
        return (patched_tensor,) + output[1:]
    return patched_tensor


def collect_forward(
    model,
    tokenizer,
    prompt: str,
    module_names: Iterable[str] = ("layer_out", "input_ln", "attn", "post_attn_ln", "mlp"),
    output_attentions: bool = True,
) -> ForwardCache:
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].tolist())
    cache: Dict[Tuple[str, int], torch.Tensor] = {}
    handles = []

    def make_hook(name: str, layer: int):
        def hook(_module, _inputs, output):
            cache[(name, layer)] = tensor_from_module_output(output).detach().cpu()
            return output

        return hook

    for layer in range(len(model.model.layers)):
        for name in module_names:
            handles.append(module_for(model, layer, name).register_forward_hook(make_hook(name, layer)))

    try:
        with torch.no_grad():
            outputs = model(
                **inputs,
                output_hidden_states=True,
                output_attentions=output_attentions,
            )
    finally:
        for handle in handles:
            handle.remove()

    attentions = None
    if output_attentions and outputs.attentions is not None:
        attentions = [attn.detach().cpu() for attn in outputs.attentions]

    return ForwardCache(
        tokens=tokens,
        input_ids=inputs["input_ids"].detach().cpu(),
        attention_mask=inputs["attention_mask"].detach().cpu(),
        module_outputs=cache,
        hidden_states=[h.detach().cpu() for h in outputs.hidden_states],
        attentions=attentions,
        logits=outputs.logits.detach().cpu(),
    )


def run_with_patch(
    model,
    tokenizer,
    target_prompt: str,
    source_cache: ForwardCache,
    patch: PatchSpec,
):
    """
    Run target_prompt while replacing one module output with source activation.

    Interpretation:
      - If patching clean numeric activation into a collapsed/corrupted run
        restores number metric, this module/layer is causally sufficient.
      - If patching collapsed activation into a clean run destroys number metric,
        this module/layer is causally necessary.
    """
    device = next(model.parameters()).device
    inputs = tokenizer(target_prompt, return_tensors="pt").to(device)
    source = source_cache.module_outputs[(patch.module_name, patch.layer)].to(device)

    def patch_hook(_module, _inputs, output):
        current = tensor_from_module_output(output).clone()
        current[:, patch.token_pos, :] = source[:, patch.token_pos, :]
        return replace_module_output(output, current)

    module = module_for(model, patch.layer, patch.module_name)
    handle = module.register_forward_hook(patch_hook)
    try:
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
    finally:
        handle.remove()

    return outputs


# ============================================================
# 4. Metrics: logit lens, attention mass, vector geometry
# ============================================================


def apply_logit_lens(model, hidden_state: torch.Tensor) -> torch.Tensor:
    """
    Convert an intermediate hidden state [hidden_dim] into vocab logits.

    LLaMA uses a final RMSNorm before lm_head, so use it for all layers.
    """
    device = next(model.parameters()).device
    h = hidden_state.to(device).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        logits = model.lm_head(model.model.norm(h))[0, 0].detach().cpu()
    return logits


def contrast_score(logits: torch.Tensor, positive_ids: List[int], negative_ids: List[int]) -> float:
    log_probs = logits.log_softmax(dim=-1)
    pos = torch.logsumexp(log_probs[positive_ids], dim=0)
    neg = torch.logsumexp(log_probs[negative_ids], dim=0)
    return (pos - neg).item()


def per_layer_logit_lens_scores(
    model,
    cache: ForwardCache,
    token_pos: int,
    positive_ids: List[int],
    negative_ids: List[int],
) -> List[float]:
    scores = []
    for h in cache.hidden_states:
        logits = apply_logit_lens(model, h[0, token_pos])
        scores.append(contrast_score(logits, positive_ids, negative_ids))
    return scores


def attention_context_mass(
    cache: ForwardCache,
    layer: int,
    query_pos: int,
    context_positions: List[int],
) -> float:
    """
    Mean attention mass from query_pos to context_positions across heads.

    For causal LMs at the number token, only previous tokens are visible.
    Define prompts so context words precede the number when testing H_attn.
    """
    if cache.attentions is None:
        raise ValueError("No attentions in cache. Set output_attentions=True.")
    attn = cache.attentions[layer][0]  # [heads, query, key]
    return attn[:, query_pos, context_positions].sum(dim=-1).mean().item()


def vector_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


def direction_projection(x: torch.Tensor, direction: torch.Tensor) -> float:
    direction = direction.float()
    direction = direction / (direction.norm() + 1e-8)
    return torch.dot(x.float().flatten(), direction.flatten()).item()


def build_direction(
    caches_a: List[ForwardCache],
    positions_a: List[int],
    caches_b: List[ForwardCache],
    positions_b: List[int],
    layer: int,
    module_name: str = "layer_out",
) -> torch.Tensor:
    """
    Mean activation difference direction.

    Example:
      numeric direction = mean(high number) - mean(low number)
      context direction = mean(medical context) - mean(nonmedical context)
    """
    vecs_a = [
        c.module_outputs[(module_name, layer)][0, pos]
        for c, pos in zip(caches_a, positions_a)
    ]
    vecs_b = [
        c.module_outputs[(module_name, layer)][0, pos]
        for c, pos in zip(caches_b, positions_b)
    ]
    return torch.stack(vecs_a).mean(dim=0) - torch.stack(vecs_b).mean(dim=0)


# ============================================================
# 5. Hypothesis tests
# ============================================================


def test_attention_hypothesis(model, tokenizer, caches: Dict[str, ForwardCache]):
    """
    H_attn:
      Prediction A: attention mass from number token to context tokens rises
                    around the same layers where number logit-lens/probe score drops.
      Prediction B: patching attn output at those layers changes the number metric
                    more than patching MLP or norm outputs.
    """
    print("\n[H_attn] attention context-mass scan")
    case = CASES[0]
    cache = caches[case.name]
    number_pos = find_number_token_span(cache.tokens, case.number)[0]

    # TODO: replace this simple heuristic with explicit context token spans.
    context_positions = list(range(1, number_pos))
    for layer in range(len(model.model.layers)):
        mass = attention_context_mass(cache, layer, number_pos, context_positions)
        print(f"  layer={layer:02d} context_mass={mass:.4f}")


def test_mlp_hypothesis(model, tokenizer, caches: Dict[str, ForwardCache]):
    """
    H_mlp:
      Prediction A: MLP output projection onto context direction increases in
                    late layers while projection onto numeric direction shrinks.
      Prediction B: patching mlp output restores/destroys number information.
    """
    print("\n[H_mlp] direction projection skeleton")
    low = caches["heart_rate_low"]
    normal = caches["heart_rate_normal"]
    low_pos = find_number_token_span(low.tokens, "24")[0]
    normal_pos = find_number_token_span(normal.tokens, "72")[0]

    for layer in range(len(model.model.layers)):
        numeric_dir = build_direction(
            caches_a=[normal],
            positions_a=[normal_pos],
            caches_b=[low],
            positions_b=[low_pos],
            layer=layer,
            module_name="mlp",
        )
        low_proj = direction_projection(low.module_outputs[("mlp", layer)][0, low_pos], numeric_dir)
        normal_proj = direction_projection(normal.module_outputs[("mlp", layer)][0, normal_pos], numeric_dir)
        print(f"  layer={layer:02d} mlp_numeric_gap={normal_proj - low_proj:+.4f}")


def test_norm_hypothesis(model, tokenizer, caches: Dict[str, ForwardCache]):
    """
    H_norm:
      Prediction A: pre/post norm activations show reduced magnitude separation
                    even when direction is partly preserved.
      Prediction B: patching input_ln or post_attn_ln has a disproportionately
                    large effect compared with nearby residual stream patching.
    """
    print("\n[H_norm] norm compression skeleton")
    low = caches["heart_rate_low"]
    normal = caches["heart_rate_normal"]
    low_pos = find_number_token_span(low.tokens, "24")[0]
    normal_pos = find_number_token_span(normal.tokens, "72")[0]

    for layer in range(len(model.model.layers)):
        pre_low = low.module_outputs[("layer_out", max(layer - 1, 0))][0, low_pos]
        pre_normal = normal.module_outputs[("layer_out", max(layer - 1, 0))][0, normal_pos]
        post_low = low.module_outputs[("input_ln", layer)][0, low_pos]
        post_normal = normal.module_outputs[("input_ln", layer)][0, normal_pos]

        pre_dist = (pre_normal - pre_low).norm().item()
        post_dist = (post_normal - post_low).norm().item()
        ratio = post_dist / (pre_dist + 1e-8)
        print(f"  layer={layer:02d} input_ln_dist_ratio={ratio:.4f}")


def patching_sweep(
    model,
    tokenizer,
    source_case: PromptCase,
    target_case: PromptCase,
    metric_fn: Callable[[torch.Tensor], float],
    modules: Iterable[str] = ("attn", "mlp", "input_ln", "post_attn_ln", "layer_out"),
):
    """
    Generic activation-patching sweep.

    metric_fn receives final logits from patched run and returns scalar score.
    Higher score should mean "more number information" or "more correct behavior".
    """
    source_cache = collect_forward(model, tokenizer, source_case.prompt)
    target_cache = collect_forward(model, tokenizer, target_case.prompt)
    target_pos = find_number_token_span(target_cache.tokens, target_case.number)[0]

    baseline_score = metric_fn(target_cache.logits)
    rows = []
    for layer in range(len(model.model.layers)):
        for module_name in modules:
            patch = PatchSpec(layer=layer, module_name=module_name, token_pos=target_pos)
            outputs = run_with_patch(model, tokenizer, target_case.prompt, source_cache, patch)
            score = metric_fn(outputs.logits.detach().cpu())
            rows.append(
                {
                    "layer": layer,
                    "module": module_name,
                    "baseline_score": baseline_score,
                    "patched_score": score,
                    "delta": score - baseline_score,
                }
            )
            print(f"  layer={layer:02d} module={module_name:12s} delta={score - baseline_score:+.4f}")
    return rows


# ============================================================
# 6. Main
# ============================================================


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer, model, _device = load_model()

    print("[1] Collecting activation caches")
    caches = {
        case.name: collect_forward(model, tokenizer, case.prompt)
        for case in CASES
    }

    print("\n[2] Logit-lens localization example")
    risk_ids = label_token_ids(tokenizer, ["critical", "dangerous", "abnormal"])
    normal_ids = label_token_ids(tokenizer, ["normal", "safe", "healthy"])
    for case in CASES[:2]:
        cache = caches[case.name]
        number_pos = find_number_token_span(cache.tokens, case.number)[0]
        scores = per_layer_logit_lens_scores(model, cache, number_pos, risk_ids, normal_ids)
        print(f"  {case.name}: " + ", ".join(f"{s:+.2f}" for s in scores))

    print("\n[3] Mechanistic hypothesis checks")
    test_attention_hypothesis(model, tokenizer, caches)
    test_mlp_hypothesis(model, tokenizer, caches)
    test_norm_hypothesis(model, tokenizer, caches)

    print("\n[4] Activation patching sweep example")

    def final_next_token_risk_metric(logits: torch.Tensor) -> float:
        # Placeholder behavioral metric at final position.
        # Replace with multi-token answer scoring for your actual prompt format.
        final_logits = logits[0, -1]
        return contrast_score(final_logits, risk_ids, normal_ids)

    patching_sweep(
        model=model,
        tokenizer=tokenizer,
        source_case=CASES[0],  # critical low heart rate
        target_case=CASES[1],  # normal heart rate
        metric_fn=final_next_token_risk_metric,
    )

    print("\nDone. Next step: replace placeholder metrics with your paper metrics.")


if __name__ == "__main__":
    main()
