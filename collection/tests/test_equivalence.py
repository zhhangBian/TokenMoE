import copy

import numpy as np

from tokenmoe_collect.equivalence import compare


def test_equivalence_checks_nondeterminism_floor_and_routing(tmp_path):
    path = tmp_path / "routing.npz"
    np.savez(
        path,
        experts=np.array([[[1, 2]], [[2, 3]]], dtype=np.uint8),
        layer_ids=[0],
        token_positions=[0, 1],
        token_ids=[5, 6, 7],
    )
    rounds = [
        [{"token_ids": [5, 6, 7], "routing_file": str(path)} for _ in range(50)] for _ in range(4)
    ]
    assert compare(rounds, "config", 128, 2)["passed"]
    changed = copy.deepcopy(rounds)
    for index in range(6):
        changed[2][index]["token_ids"] = [99]
    changed[1][0]["token_ids"] = [98]
    result = compare(changed, "config", 128, 2)
    assert not result["passed"] and result["off_off_disagreements"] == 1
    invalid = tmp_path / "invalid.npz"
    np.savez(
        invalid,
        experts=np.array([[[1, 1]], [[2, 3]]], dtype=np.uint8),
        layer_ids=[0],
        token_positions=[0, 1],
        token_ids=[5, 6, 7],
    )
    rounds[2][0]["routing_file"] = str(invalid)
    result = compare(rounds, "config", 128, 2)
    assert result["invalid_routing"] and result["routing_disagreements"]
