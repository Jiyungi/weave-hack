"""Property-based tests for the GPT_Curation_Node (Track B / Req 12).

Covers:

* task 5.6 — Property 15: curation output conforms to the Training_Pair schema
* task 5.7 — Property 16: curation conserves and accounts for every interaction

A controllable mock curator tags each interaction as curable / malformed /
uncurable so the ground-truth discard count is known. Each property runs a
minimum of 100 generated cases via Hypothesis.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from weaveself.contracts.training_pair import TrainingPair
from weaveself.data.curation import GPTCurationNode

_unit_labels = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=6
)


class TaggedCurator:
    """A mock curator whose behavior is driven by each interaction's ``kind``.

    * ``"curable"`` -> a valid pair (with a deliberately wrong label, to verify
      the node re-stamps the correct ``unit_label``);
    * ``"malformed"`` -> a mapping missing ``completion`` (node must discard);
    * anything else -> ``None`` (uncurable).
    """

    def curate(self, interaction, unit_label):
        kind = interaction.get("kind")
        if kind == "curable":
            return {
                "prompt": interaction["prompt"],
                "completion": interaction["completion"],
                "unit_label": "DELIBERATELY_WRONG",
            }
        if kind == "malformed":
            return {"prompt": interaction["prompt"]}  # missing completion
        return None


@st.composite
def _interactions(draw, *, max_n: int = 20):
    n = draw(st.integers(min_value=0, max_value=max_n))
    items = []
    for i in range(n):
        kind = draw(st.sampled_from(["curable", "malformed", "uncurable"]))
        items.append({"kind": kind, "prompt": f"p{i}", "completion": f"c{i}"})
    return items


# Feature: weaveself, Property 15: Curation output conforms to the Training_Pair schema
@settings(max_examples=100)
@given(interactions=_interactions(), unit_label=_unit_labels)
def test_property_15_schema_conformance(interactions, unit_label):
    node = GPTCurationNode(TaggedCurator())
    result = node.curate_interactions(interactions, unit_label)

    for pair in result.pairs:
        # Every emitted pair is a schema-valid Training_Pair (Req 12.1, 12.3).
        assert isinstance(pair, TrainingPair)
        assert isinstance(pair.prompt, str) and pair.prompt
        assert isinstance(pair.completion, str) and pair.completion
        # And it carries the correct Unit's label, not the curator's wrong one.
        assert pair.unit_label == unit_label


# Feature: weaveself, Property 16: Curation conserves and accounts for every interaction
@settings(max_examples=100)
@given(interactions=_interactions(), unit_label=_unit_labels)
def test_property_16_count_conservation(interactions, unit_label):
    node = GPTCurationNode(TaggedCurator())
    result = node.curate_interactions(interactions, unit_label)

    curable = sum(1 for i in interactions if i["kind"] == "curable")
    uncurable_or_malformed = len(interactions) - curable

    # Emitted + discarded accounts for every input interaction (Req 12.4).
    assert result.emitted + result.discarded == len(interactions)
    # Discarded equals exactly the interactions yielding no valid pair (Req 12.4).
    assert result.emitted == curable
    assert result.discarded == uncurable_or_malformed
