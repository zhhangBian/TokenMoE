"""Locate harness content in the engine's decoded prompt, without re-rendering it."""

import bisect

from tokenizers.decoders import DecodeStream

from .ids import content_id


def decode_with_offsets(tokenizer, token_ids):
    text = tokenizer.decode(token_ids, skip_special_tokens=False)
    stream = DecodeStream(skip_special_tokens=False)
    decoded, starts = "", []
    for token in token_ids:
        starts.append(len(decoded))
        decoded += stream.step(tokenizer, int(token)) or ""
    if not text.startswith(decoded):
        # Some decoders clean earlier whitespace. Full prefixes are the fallback.
        starts = [
            len(tokenizer.decode(token_ids[:i], skip_special_tokens=False).rstrip("\ufffd"))
            for i in range(len(token_ids))
        ]
        starts = [min(start, len(text)) for start in starts]
        if starts != sorted(starts):
            raise ValueError("Tokenizer does not provide monotonic decoded character offsets")
    return text, starts


def locate(text, value, cursor):
    found = text.find(value, cursor)
    if found >= 0:
        return found, found + len(value), list(range(found, found + len(value) + 1))
    query = [(i, c) for i, c in enumerate(value) if not c.isspace()]
    hay = [(i, c) for i, c in enumerate(text[cursor:], cursor) if not c.isspace()]
    needle = "".join(c for _, c in query)
    if not needle:
        return None
    offset = "".join(c for _, c in hay).find(needle)
    if offset < 0:
        return None
    positions = [i for i, _ in hay[offset : offset + len(needle)]]
    source_positions = [i for i, _ in query]
    mapping = []
    for boundary in range(len(value) + 1):
        index = bisect.bisect_left(source_positions, boundary)
        mapping.append(positions[index] if index < len(positions) else positions[-1] + 1)
    return positions[0], positions[-1] + 1, mapping


def align_prompt(tokenizer, token_ids, messages, provenance, request_id):
    text, token_starts = decode_with_offsets(tokenizer, token_ids)
    located, failed = [], []
    cursor = 0
    for index, (message, source) in enumerate(zip(messages, provenance)):
        fields = []
        if message.get("reasoning"):
            fields.append(("reasoning", message["reasoning"]))
        if isinstance(message.get("content"), str) and message["content"]:
            fields.append(("content", message["content"]))
        for field, value in fields:
            base = {
                "llm_request_id": request_id,
                "message_index": index,
                "message_field": field,
                "segment_type": source["segment_type"],
                "source_tool_call_id": source.get("source_tool_call_id"),
                "source_session_id": source.get("source_session_id"),
                "produced_at": source.get("produced_at"),
                "available_at": source.get("available_at"),
                "alignment_error": None,
            }
            match = locate(text, value, cursor)
            if match is None:
                failed.append(
                    base
                    | {
                        "char_start": None,
                        "char_end": None,
                        "token_start": None,
                        "token_end": None,
                        "alignment_error": "message_not_found",
                    }
                )
                continue
            a, b, mapping = match
            cursor = b
            if source["segment_type"] == "tool_output" and field == "content":
                point = a
                for left, right in source.get("payload_ranges", []):
                    x, y = mapping[left], mapping[right]
                    if point < x:
                        located.append(
                            base
                            | {
                                "char_start": point,
                                "char_end": x,
                                "segment_type": "harness_scaffold",
                                "source_tool_call_id": None,
                            }
                        )
                    if x < y:
                        located.append(base | {"char_start": x, "char_end": y})
                    point = y
                if point < b:
                    located.append(
                        base
                        | {
                            "char_start": point,
                            "char_end": b,
                            "segment_type": "harness_scaffold",
                            "source_tool_call_id": None,
                        }
                    )
            else:
                located.append(base | {"char_start": a, "char_end": b})
    segments, cursor = [], 0
    for segment in sorted(located, key=lambda s: s["char_start"]):
        if segment["char_start"] < cursor:
            raise ValueError("Located prompt segments overlap")
        if cursor < segment["char_start"]:
            segments.append(
                {
                    "llm_request_id": request_id,
                    "segment_type": "other",
                    "source_tool_call_id": None,
                    "source_session_id": None,
                    "produced_at": None,
                    "available_at": None,
                    "message_index": None,
                    "message_field": None,
                    "char_start": cursor,
                    "char_end": segment["char_start"],
                    "alignment_error": None,
                }
            )
        segments.append(segment)
        cursor = segment["char_end"]
    if cursor < len(text):
        segments.append(
            {
                "llm_request_id": request_id,
                "segment_type": "other",
                "source_tool_call_id": None,
                "source_session_id": None,
                "produced_at": None,
                "available_at": None,
                "message_index": None,
                "message_field": None,
                "char_start": cursor,
                "char_end": len(text),
                "alignment_error": None,
            }
        )
    for segment in segments:
        segment["token_start"] = bisect.bisect_left(token_starts, segment["char_start"])
        segment["token_end"] = bisect.bisect_left(token_starts, segment["char_end"])
    segments.extend(failed)
    for i, segment in enumerate(segments):
        segment["segment_id"] = "seg_" + content_id([request_id, i, segment])
    total = sum(
        bool(m.get("reasoning")) + bool(isinstance(m.get("content"), str) and m["content"])
        for m in messages
    )
    return (
        text,
        segments,
        {"message_fields": total, "located": total - len(failed), "failed": len(failed)},
    )
