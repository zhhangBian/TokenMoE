"""Real-checkpoint forced-input shard probe, separate from free generation."""

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch
from safetensors import safe_open
from vllm.config.expert_cache import ExpertCacheConfig
from vllm.model_executor.layers.fused_moe.expert_cache import ExpertCache
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
from vllm.model_executor.layers.fused_moe.router.fused_topk_router import fused_topk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    model = Path("/home/youwei/bzh/model/Qwen/Qwen3-30B-A3B")
    index = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]

    def weight(key):
        with safe_open(model / index[key], framework="pt", device="cpu") as reader:
            return reader.get_tensor(key)

    prefix = "model.layers.0.mlp"
    gate = weight(prefix + ".gate.weight").cuda()
    host13, host2 = [], []
    for expert in range(128):
        key = f"{prefix}.experts.{expert}"
        host13.append(
            torch.cat([weight(key + ".gate_proj.weight"), weight(key + ".up_proj.weight")])
        )
        host2.append(weight(key + ".down_proj.weight"))
    full13, full2 = torch.stack(host13), torch.stack(host2)
    torch.manual_seed(42)
    x = torch.randn(32, 2048, dtype=torch.bfloat16, device="cuda") * 0.2
    logits = torch.nn.functional.linear(x, gate).float()
    coefficients, ids, _ = fused_topk(x, logits, 8, True)
    result = {
        "schema_version": 1,
        "scope": "real Qwen layer 0; forced identical inputs and actual router",
        "criteria": {"atol": 0.02, "rtol": 0.02},
        "devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "status": "running",
        "shards": [],
    }

    def save():
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")

    save()
    try:
        for tp in (1, 2):
            width = 768 // tp
            for rank in range(tp):
                w13 = torch.cat(
                    [
                        full13[:, rank * width : (rank + 1) * width],
                        full13[:, 768 + rank * width : 768 + (rank + 1) * width],
                    ],
                    dim=1,
                ).contiguous()
                w2 = full2[:, :, rank * width : (rank + 1) * width].contiguous()
                cache = ExpertCache(
                    w13,
                    w2,
                    ExpertCacheConfig(enabled=True, slots_per_layer=16),
                    torch.device("cuda", 0),
                )

                def apply(*, layer, x, topk_weights, topk_ids, **kwargs):
                    return fused_experts(
                        x, layer.w13_weight, layer.w2_weight, topk_weights, topk_ids
                    )

                layer = SimpleNamespace(
                    w13_weight=cache.weights[0],
                    w2_weight=cache.weights[1],
                    moe_config=SimpleNamespace(tp_size=1),
                    quant_method=SimpleNamespace(apply=apply),
                )
                reference = fused_experts(x.clone(), w13.cuda(), w2.cuda(), coefficients, ids)
                output = cache.execute(layer, x.clone(), coefficients, ids)
                error = (output.float() - reference.float()).abs()
                torch.testing.assert_close(output, reference, atol=0.02, rtol=0.02)
                result["shards"].append(
                    {
                        "tp": tp,
                        "rank": rank,
                        "max_abs_error": error.max().item(),
                        "mean_abs_error": error.mean().item(),
                        "tokens": len(x),
                        "unique_experts": ids.unique().numel(),
                        "counters": dict(cache.counters),
                    }
                )
                cache.close()
                del layer, cache, reference, output, w13, w2
        result["status"] = "passed-layer-shard-probe"
        save()
    except BaseException as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        save()
        raise


if __name__ == "__main__":
    main()
