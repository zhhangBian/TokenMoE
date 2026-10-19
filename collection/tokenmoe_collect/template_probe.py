"""CPU-only template evidence, executed with the serving tokenizer's environment."""

import copy
import hashlib
import json
import sys
from pathlib import Path


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def probe(root):
    import numpy as np
    from transformers import AutoTokenizer

    root = Path(root)
    profile = json.loads((root / "static/model_profiles.jsonl").read_text().splitlines()[0])
    model = json.loads((root / "model_config.json").read_text())
    content = json.loads(next((root / "static/role_templates").rglob("config.json")).read_text())
    tokenizer = AutoTokenizer.from_pretrained(model["model_path"], local_files_only=True)
    template_hash = hashlib.sha256(
        (Path(model["model_path"]) / "tokenizer_config.json").read_bytes()
    ).hexdigest()
    source = json.loads((root / "engine_source.json").read_text())
    raw = [
        json.loads(line)
        for path in sorted((root / "raw/harness").glob("*/llm_requests.jsonl"))
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    reconstructed, result = {}, {}
    engine_requests = {
        row["llm_request_id"]: row
        for row in (
            json.loads(line)
            for line in (Path(source["path"]) / "requests.jsonl").read_text().splitlines()
        )
    }
    for request in sorted(raw, key=lambda r: (r["request_created_at"], r["llm_request_id"])):
        rid = request["llm_request_id"]
        base = (
            reconstructed[request["messages_base_request_id"]]
            if request["messages_mode"] == "delta"
            else []
        )
        messages = base + request["messages_delta"]
        reconstructed[rid] = messages
        route = Path(source["path"]) / "routing" / f"{rid}.npz"
        if not route.exists():
            continue
        with np.load(route, allow_pickle=False) as data:
            # Read the actual prompt prefix, independent of sampled output.
            prompt = data["token_ids"].tolist()
        engine_request = engine_requests.get(rid)
        if engine_request is None:
            continue
        prompt = prompt[: engine_request["num_prompt_tokens"]]
        prepared = copy.deepcopy(messages)
        # Match vLLM's assistant-message field adapter for this pinned fork.
        for message in prepared:
            if message["role"] == "assistant" and message.get("reasoning") is not None:
                message["reasoning_content"] = message["reasoning"]
        kwargs = {
            "tools": content["tools"],
            "add_generation_prompt": True,
            "tokenize": True,
            "return_dict": False,
            **model.get("sampling", {}).get("chat_template_kwargs", {}),
        }
        rendered = tokenizer.apply_chat_template(prepared, **kwargs)
        if rendered != prompt:
            result[rid] = {"full_render_matches_engine": False, "omitted_fields": []}
            continue
        omitted = []
        for index, message in enumerate(prepared):
            if not message.get("reasoning"):
                continue
            ablated = copy.deepcopy(prepared)
            ablated[index].pop("reasoning", None)
            ablated[index].pop("reasoning_content", None)
            if tokenizer.apply_chat_template(ablated, **kwargs) == prompt:
                omitted.append(
                    {
                        "message_index": index,
                        "message_field": "reasoning",
                        "template_sha256": template_hash,
                        "messages_sha256": digest(messages),
                        "engine_prompt_ids_sha256": digest(prompt),
                        "full_render_matches_engine": True,
                        "ablated_render_unchanged": True,
                        "tokenizer_sha256": profile["tokenizer_hash"],
                    }
                )
        result[rid] = {"full_render_matches_engine": True, "omitted_fields": omitted}
    return result


if __name__ == "__main__":
    output = Path(sys.argv[2])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(probe(sys.argv[1]), indent=2) + "\n")
