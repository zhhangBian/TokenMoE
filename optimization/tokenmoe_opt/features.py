"""Causal features; historical label semantics remain separately reproducible."""

import shlex
from pathlib import Path

from tokenmoe_collect.report import tool_category

FEATURE_VERSION = "request-features-v2"
LABEL_VERSION = "shell-command-v2"
HISTORICAL_LABEL_VERSION = "command-outcome-v1"
COMMANDS = {
    "test": {"pytest", "unittest", "tox", "nosetests", "ctest"},
    "inspect": {"grep", "rg", "find", "ls", "head", "tail", "cat", "sed", "pwd", "git"},
    "edit": {"patch", "apply_patch", "tee"},
    "build_or_install": {"pip", "pip3", "conda", "apt", "apt-get", "make", "cmake", "build"},
}


def command_categories(command, version=LABEL_VERSION):
    if version == HISTORICAL_LABEL_VERSION:
        return (tool_category(command),)
    if version != LABEL_VERSION:
        raise ValueError(f"Unknown shell label rules: {version}")
    # Inspect executable words, never payloads in quoted strings or heredocs.
    if "<<" in command or "$(" in command or "`" in command:
        return ("unknown",)
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|<>")
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError:
        return ("unknown",)
    categories, index, executable = set(), 0, True
    while index < len(tokens):
        word = tokens[index]
        if word in {";", "&&", "||", "|", "&"}:
            executable = True
        elif word in {">", ">>"}:
            categories.add("edit")
            index += 1
        elif executable:
            if "=" in word and not word.startswith("/"):
                index += 1
                continue
            name = Path(word).name
            if name in {"env", "sudo", "command"}:
                index += 1
                continue
            if name in {"python", "python3"} and tokens[index + 1 : index + 2] == ["-m"]:
                name = tokens[index + 2] if index + 2 < len(tokens) else "unknown"
            found = {key for key, commands in COMMANDS.items() if name in commands}
            categories.update(found or {"unknown"})
            executable = False
        index += 1
    return tuple(sorted(categories or {"unknown"}))


def request_features(
    request, tools, root, *, first_request, previous_key=None, label_version=LABEL_VERSION
):
    decision = request["request_created_at"]
    categories, availability, outcomes = [], [], []
    ids = request["prev_tool_call_ids"]
    for identifier in ids:
        tool = tools[identifier]
        parsed = tool.get("parsed_at") or tool.get("started_at")
        finished = tool.get("finished_at")
        if parsed is None or finished is None or max(parsed, finished) > decision:
            raise ValueError(f"Unavailable preceding tool at request decision: {identifier}")
        command = (root / tool["tool_args_ref"]).read_text()
        categories.extend(command_categories(command, label_version))
        # A consumer request proves observation; historical transport delay is unknown.
        availability.append(
            {
                "tool_call_id": identifier,
                "source_observed_at": parsed,
                "receipt_at": decision,
                "available_at": decision,
                "receipt_provenance": "historical-consuming-request-upper-bound",
                "outcome_observed_at": finished,
            }
        )
        outcomes.append(
            "timeout"
            if tool.get("timed_out")
            else "success"
            if tool.get("exit_status") == 0
            else "failure"
            if tool.get("exit_status") is not None
            else "unknown"
        )
    key = tuple(sorted(set(categories)))
    return {
        "version": FEATURE_VERSION,
        "label_version": label_version,
        "decision_at": decision,
        "first_request": first_request,
        "stage": "first" if first_request else "subsequent",
        "no_tool": not ids,
        "multiple_tools": len(ids) > 1,
        "tool_key": key,
        "unknown_category": "unknown" in key,
        "tool_switch": previous_key is not None and bool(key) and key != previous_key,
        "observed_outcomes": outcomes,
        "availability": availability,
        "visible_event_sequence": None,
        "sequence_reason": "Historical finalized traces have no controller event sequence",
    }
