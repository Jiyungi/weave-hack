"""Property-based and example tests for the Weave_Eval (Track B / Req 14, 15).

Property tests (min 100 generated cases via Hypothesis) feed generated numeric
score matrices and label sets into the pure decision and matrix-construction
functions (per the design's cost-control note):

* task 7.3 — Property 20: personalization pass decision
* task 7.4 — Property 21: competitive comparison pass decision
* task 7.5 — Property 22: predicted unit is the minimum-perplexity adapter
* task 7.6 — Property 23: confusion matrix is well-formed
* task 7.7 — Property 24: eval artifact conforms to the schema

An example test covers the context-memory baseline being computed and recorded
for a representative Unit (Req 14.3) using a deterministic mock score function.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from weaveself.contracts.eval_results import (
    EvalResults,
    validate_eval_results,
)
from weaveself.eval.weave_eval import (
    HeldOutSet,
    WeaveEval,
    build_confusion_matrix,
    competitive_pass,
    confusion_from_scores,
    personalization_pass,
    predicted_unit,
    record_size_bytes,
)
from weaveself.contracts.training_pair import TrainingPair

_perplexity = st.floats(
    min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False
)
_labels = st.lists(
    st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=5),
    min_size=1,
    max_size=6,
    unique=True,
)


# Feature: weaveself, Property 20: Personalization pass decision
@settings(max_examples=100)
@given(adapter=_perplexity, base=_perplexity)
def test_property_20_personalization_pass(adapter, base):
    # Passes if and only if adapter perplexity is strictly below base (Req 14.2).
    assert personalization_pass(adapter, base) == (adapter < base)


# Feature: weaveself, Property 21: Competitive comparison pass decision
@settings(max_examples=100)
@given(adapter=_perplexity, baseline=_perplexity)
def test_property_21_competitive_pass(adapter, baseline):
    # Passes iff adapter perplexity is <= the context-memory baseline (Req 14.4).
    assert competitive_pass(adapter, baseline) == (adapter <= baseline)


# Feature: weaveself, Property 22: Predicted unit is the minimum-perplexity adapter
@settings(max_examples=100)
@given(data=st.data())
def test_property_22_predicted_is_argmin(data):
    labels = data.draw(_labels)
    scores = {
        label: data.draw(_perplexity, label=f"ppl-{label}") for label in labels
    }
    chosen = predicted_unit(scores)
    min_ppl = min(scores.values())
    # The predicted unit achieves the minimum perplexity (Req 15.1).
    assert scores[chosen] == min_ppl
    # Ties broken by order: it is the first label achieving the minimum.
    first_min = next(l for l in labels if scores[l] == min_ppl)
    assert chosen == first_min


# Feature: weaveself, Property 23: Confusion matrix is well-formed
@settings(max_examples=100)
@given(data=st.data())
def test_property_23_confusion_well_formed(data):
    labels = data.draw(_labels)
    n_predictions = data.draw(st.integers(min_value=1, max_value=20))
    predictions = [
        (
            data.draw(st.sampled_from(labels), label=f"true-{i}"),
            data.draw(st.sampled_from(labels), label=f"pred-{i}"),
        )
        for i in range(n_predictions)
    ]
    cm = build_confusion_matrix(labels, predictions)

    n = len(labels)
    # Square with dimensions equal to the label count (Req 15.2, 5.3).
    assert len(cm.matrix) == n
    assert all(len(row) == n for row in cm.matrix)
    # Each held-out set contributes exactly one count; total == #predictions.
    total = sum(sum(row) for row in cm.matrix)
    assert total == float(n_predictions)
    # Each prediction lands in cell [true][predicted].
    index = {label: i for i, label in enumerate(labels)}
    expected = [[0.0] * n for _ in range(n)]
    for true_label, pred_label in predictions:
        expected[index[true_label]][index[pred_label]] += 1.0
    assert cm.matrix == expected


def test_confusion_from_scores_picks_argmin_per_row():
    labels = ["alice", "bob"]
    score_rows = {
        "alice": {"alice": 5.0, "bob": 9.0},  # alice's set best under alice adapter
        "bob": {"alice": 8.0, "bob": 4.0},     # bob's set best under bob adapter
    }
    cm = confusion_from_scores(labels, score_rows)
    # Perfect diagonal: each Unit identified by its own adapter (Req 15.1, 15.2).
    assert cm.matrix == [[1.0, 0.0], [0.0, 1.0]]


# Feature: weaveself, Property 24: Eval artifact conforms to the schema
@settings(max_examples=100)
@given(data=st.data())
def test_property_24_eval_artifact_schema(data):
    labels = data.draw(_labels)
    predictions = [(label, label) for label in labels]
    confusion = build_confusion_matrix(labels, predictions)

    results = EvalResults(
        perplexity={
            "base": data.draw(_perplexity, label="base"),
            "adapter": data.draw(_perplexity, label="adapter"),
            "context_memory": data.draw(_perplexity, label="ctx"),
        },
        confusion_matrix=confusion,
        size_bytes=record_size_bytes(
            data.draw(st.integers(min_value=0, max_value=200_000), label="nkt"),
            data.draw(st.integers(min_value=0, max_value=10_000_000), label="lora"),
        ),
        examples=[
            {
                "prompt": "p",
                "base": "b",
                "adapter": "a",
                "reference": "r",
            }
        ],
    )

    # Round-trips through validate without error and keeps the schema (Req 5).
    revalidated = validate_eval_results(results.model_dump())
    assert isinstance(revalidated, EvalResults)
    assert isinstance(revalidated.perplexity.base, float)
    assert isinstance(revalidated.perplexity.adapter, float)
    assert isinstance(revalidated.perplexity.context_memory, float)
    assert len(revalidated.confusion_matrix.labels) >= 1
    assert len(revalidated.confusion_matrix.matrix) == len(
        revalidated.confusion_matrix.labels
    )
    assert isinstance(revalidated.size_bytes.nktmirror, int)
    assert isinstance(revalidated.size_bytes.lora, int)
    for ex in revalidated.examples:
        assert isinstance(ex.prompt, str)
        assert isinstance(ex.base, str)
        assert isinstance(ex.adapter, str)
        assert isinstance(ex.reference, str)


# --- Example test: context-memory baseline computed and recorded (Req 14.3) ---


def test_context_memory_baseline_recorded():
    rows = [
        TrainingPair(prompt="how are you?", completion="great", unit_label="alice"),
        TrainingPair(prompt="weekend plan?", completion="hiking", unit_label="alice"),
    ]

    def score_fn(prompt, target, adapter_id):
        # Adapter beats base; injecting context (longer prompt) helps base a bit.
        if adapter_id is not None:
            return 5.0
        # Context-memory prompts are prefixed, so they are longer than the bare
        # held-out prompt; reward that with a slightly lower base perplexity.
        return 9.0 if "\n" in prompt else 12.0

    heldout = HeldOutSet(
        unit_label="alice",
        rows=rows,
        context_examples=["alice likes hiking", "alice is upbeat"],
    )
    unit_eval = WeaveEval(score_fn).evaluate_unit(heldout, adapter_id="alice-d0")

    assert unit_eval.adapter_perplexity == 5.0
    assert unit_eval.base_perplexity == 12.0
    # The context-memory baseline was scored on the same set and recorded (14.3).
    assert unit_eval.context_memory_perplexity == 9.0
    assert unit_eval.personalization_passed is True  # 5 < 12
    assert unit_eval.competitive_passed is True       # 5 <= 9
