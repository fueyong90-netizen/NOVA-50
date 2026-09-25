#!/usr/bin/env python3
"""
Computes REAL trainable-parameter counts and rough memory estimates for a
set of candidate NOVA-50 architectures, so a final choice can be made on
evidence rather than "closer to 50M = better".

This does NOT train the candidates (that requires real data + GPU, not
available in this environment — see docs/COMPETITION_RULES.md). It only
performs the parameter/memory arithmetic, which is exact and requires no
training. Selecting a final candidate based on benchmark/perplexity
performance requires running scripts/run_experiment.py on each candidate
once real training is possible — see results/architecture_search.md for
what remains NOT RUN.
"""
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.layers import ModelConfig
from model.transformer import build_model

BUDGET = 50_000_000

# Candidates explored: varying dim/n_layers/ffn ratio around the existing
# NOVA-50 (32.9M) up toward the budget ceiling, all using the same
# architectural family (RMSNorm + RoPE + GQA + SwiGLU + weight tying) as
# configs/model_nova.yaml, so the comparison isolates SIZE, not architecture.
CANDIDATES = [
    {"label": "nova50_current_32.9M", "vocab_size": 8000, "dim": 512, "n_layers": 10,
     "n_heads": 8, "n_kv_heads": 4, "ffn_hidden_dim": 1365, "max_seq_len": 512},
    {"label": "nova50_wider_40M", "vocab_size": 8000, "dim": 576, "n_layers": 10,
     "n_heads": 9, "n_kv_heads": 3, "ffn_hidden_dim": 1536, "max_seq_len": 512},
    {"label": "nova50_deeper_42M", "vocab_size": 8000, "dim": 512, "n_layers": 14,
     "n_heads": 8, "n_kv_heads": 4, "ffn_hidden_dim": 1365, "max_seq_len": 512},
    {"label": "nova50_wide_deep_45M", "vocab_size": 8000, "dim": 576, "n_layers": 12,
     "n_heads": 9, "n_kv_heads": 3, "ffn_hidden_dim": 1536, "max_seq_len": 512},
    {"label": "nova50_max_48M", "vocab_size": 8000, "dim": 640, "n_layers": 11,
     "n_heads": 10, "n_kv_heads": 5, "ffn_hidden_dim": 1706, "max_seq_len": 512},
    {"label": "nova50_ceiling_49M", "vocab_size": 8000, "dim": 640, "n_layers": 12,
     "n_heads": 10, "n_kv_heads": 5, "ffn_hidden_dim": 1600, "max_seq_len": 512},
]

FIXED = dict(dropout=0.0, norm_type="rmsnorm", ffn_type="swiglu", use_rope=True,
             weight_tying=True, param_sharing=False, shared_group_size=1, init_std=0.02)


def estimate_memory_gb(n_params: int, optimizer_states: int = 2, bytes_per_param: int = 4) -> dict:
    """Very rough estimate: params + grads + optimizer states, fp32, no activations."""
    weights = n_params * bytes_per_param
    grads = n_params * bytes_per_param
    optim = n_params * bytes_per_param * optimizer_states  # AdamW: 2 states (m, v)
    total_bytes = weights + grads + optim
    return {"model_gb": weights / 1e9, "optimizer_states_gb": optim / 1e9,
            "grads_gb": grads / 1e9, "total_gb_fp32_no_activations": total_bytes / 1e9}


def main():
    rows = []
    for c in CANDIDATES:
        label = c.pop("label")
        cfg = ModelConfig(**c, **FIXED)
        model = build_model(cfg)
        counts = model.count_parameters()
        total = counts["total_trainable"]
        mem = estimate_memory_gb(total)
        rows.append({
            "label": label, "config": {**c, **FIXED},
            "total_trainable_parameters": total,
            "percentage_of_budget": round(100 * total / BUDGET, 2),
            "status": "PASS" if total <= BUDGET else "FAIL",
            "estimated_memory_fp32_gb_no_activations": round(mem["total_gb_fp32_no_activations"], 3),
        })
        c["label"] = label  # restore for printing

    print(f"{'label':<24} {'params':>12} {'%budget':>8} {'status':>6} {'est.mem(GB)':>12}")
    for r in rows:
        print(f"{r['label']:<24} {r['total_trainable_parameters']:>12,} "
              f"{r['percentage_of_budget']:>7.2f}% {r['status']:>6} "
              f"{r['estimated_memory_fp32_gb_no_activations']:>12.3f}")

    out_json = Path("results/architecture_search.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(rows, f, indent=2)

    out_md = Path("results/architecture_search.md")
    with open(out_md, "w") as f:
        f.write("# Architecture Search — parameter/memory arithmetic (NOT a training result)\n\n")
        f.write("All rows use the same NOVA-50 architectural family (RMSNorm, RoPE, GQA, "
                "SwiGLU, weight tying) as `configs/model_nova.yaml`; only size varies, so this "
                "isolates the effect of scale. Parameter counts and memory estimates below are "
                "EXACT arithmetic (not measured on hardware). Benchmark/perplexity performance "
                "for each candidate is **NOT RUN** — that requires real training data and a GPU, "
                "not available in this development environment (see `docs/COMPETITION_RULES.md`). "
                "The current submission keeps `nova50_current_32.9M` as the selected architecture "
                "because it is the only candidate that has actually been exercised end-to-end "
                "(demo training, tests) in this repository; picking a bigger candidate without "
                "evidence it trains/evaluates better would contradict the \"no fake results\" rule.\n\n")
        f.write(f"| label | params | % of 50M budget | status | est. fp32 memory (GB, no activations) |\n")
        f.write("|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['label']} | {r['total_trainable_parameters']:,} | "
                    f"{r['percentage_of_budget']}% | {r['status']} | "
                    f"{r['estimated_memory_fp32_gb_no_activations']} |\n")
        f.write("\n## NOT RUN for every candidate above\n"
                "- Actual training loss / validation loss\n- Perplexity\n"
                "- HellaSwag / ARC-Easy / PIQA / WinoGrande\n- WikiText-103 perplexity\n"
                "- Wall-clock training time on real hardware\n")

    print(f"\n-> {out_json}\n-> {out_md}")


if __name__ == "__main__":
    main()
