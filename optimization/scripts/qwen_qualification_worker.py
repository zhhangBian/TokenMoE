"""Opt-in vLLM worker extension for matched full-model tensor diagnostics."""

import os
import time
from pathlib import Path

import numpy as np
import psutil
import torch
from vllm.forward_context import get_forward_context


class QwenQualificationWorker:
    def qualification_state(self):
        caches = [
            m.expert_cache
            for m in self.model_runner.get_model().modules()
            if hasattr(m, "expert_cache")
        ]
        return {
            "rank": self.rank,
            "pid": os.getpid(),
            "create_time": psutil.Process().create_time(),
            "torch_threads": torch.get_num_threads(),
            "omp_threads": os.environ.get("OMP_NUM_THREADS"),
            "delayed_loads": sum(c.counters.get("qualification_delayed_loads", 0) for c in caches),
        }

    def delay_qualification_shard(self, rank=1, seconds=0.05):
        if not 0 < seconds <= 0.2:
            raise ValueError("Qualification delay must be within (0, 0.2] seconds")
        if self.rank == rank:
            for module in self.model_runner.get_model().modules():
                if not hasattr(module, "expert_cache"):
                    continue
                cache, original, delayed = module.expert_cache, module.expert_cache._load, [False]

                def load(
                    expert,
                    protected,
                    *,
                    trigger="demand",
                    _cache=cache,
                    _original=original,
                    _delayed=delayed,
                ):
                    if trigger == "demand" and expert not in _cache.resident and not _delayed[0]:
                        _delayed[0] = True
                        _cache.counters["qualification_delayed_loads"] += 1
                        time.sleep(seconds)
                    return _original(expert, protected, trigger=trigger)

                cache._load = load
        return {"rank": self.rank, "delay_seconds": seconds if self.rank == rank else 0}

    def arm_qualification_peer_failure(self, rank=1):
        if self.rank == rank:
            for module in self.model_runner.get_model().modules():
                if not hasattr(module, "expert_cache"):
                    continue
                cache, original = module.expert_cache, module.expert_cache._load

                def load(expert, protected, *, trigger="demand", _cache=cache, _original=original):
                    if trigger == "demand" and expert not in _cache.resident:
                        os._exit(73)
                    return _original(expert, protected, trigger=trigger)

                cache._load = load
        return {"rank": self.rank, "armed": self.rank == rank}

    def enable_expert_probes(self, directory: str, frames: int = 3):
        from vllm.config.compilation import CUDAGraphMode

        cc = self.vllm_config.compilation_config
        split_moe = cc.cudagraph_mode == CUDAGraphMode.PIECEWISE and "vllm::moe_forward" in (
            cc.splitting_ops or []
        )
        if not (self.model_config.enforce_eager or split_moe) or not 1 <= frames <= 8:
            raise ValueError(
                "Expert probes require eager or split-MoE piecewise execution and 1..8 frames"
            )
        destination = Path(directory) / f"rank-{self.rank}"
        destination.mkdir(parents=True, exist_ok=False)
        count = 0
        for module in self.model_runner.get_model().modules():
            if not hasattr(module, "forward_modular"):
                continue
            original = module.forward_modular
            index, calls = count, [0]

            def probe(*args, _original=original, _index=index, _calls=calls, **kwargs):
                call = _calls[0]
                _calls[0] += 1
                if call >= frames:
                    return _original(*args, **kwargs)
                x = kwargs["x"].detach().clone()
                ids = kwargs["topk_ids"].detach().clone()
                coefficients = kwargs["topk_weights"].detach().clone()
                padding = get_forward_context().is_padding
                output = _original(*args, **kwargs)
                if not isinstance(output, torch.Tensor):
                    raise TypeError("Expected a modular tensor output")
                np.savez(
                    destination / f"layer-{_index:02d}-frame-{call:02d}.npz",
                    x=x.float().cpu().numpy(),
                    ids=ids.cpu().numpy(),
                    coefficients=coefficients.float().cpu().numpy(),
                    output=output.detach().float().cpu().numpy(),
                    padding=(
                        padding[: len(x)].cpu().numpy()
                        if padding is not None
                        else np.zeros(len(x), dtype=bool)
                    ),
                )
                return output

            module.forward_modular = probe
            count += 1
        return {
            "rank": self.rank,
            "layers": count,
            "frames": frames,
            "scope": "diagnostic CPU copies; timings include probe cost",
        }
