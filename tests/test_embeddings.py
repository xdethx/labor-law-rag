"""Offline tests for the FlagEmbedding -> Qdrant sparse-format bridge.

`to_indices_values` is a pure function; no model load, no qdrant, no network.
"""

import numpy as np

from src.embeddings import to_indices_values


def test_string_token_ids_become_int_indices():
    indices, values = to_indices_values({"1293": 0.5, "7": 0.25})
    assert indices == [1293, 7]
    assert values == [0.5, 0.25]


def test_numpy_weights_become_plain_floats():
    # FlagEmbedding emits numpy scalars; Qdrant serialization needs floats.
    indices, values = to_indices_values({"42": np.float16(0.375)})
    assert indices == [42]
    assert all(type(v) is float for v in values)
    assert values == [0.375]


def test_index_value_pairing_preserved():
    weights = {"10": 0.1, "20": 0.2, "30": 0.3}
    indices, values = to_indices_values(weights)
    assert dict(zip(indices, values)) == {10: 0.1, 20: 0.2, 30: 0.3}


def test_empty_weights_give_empty_lists():
    assert to_indices_values({}) == ([], [])
