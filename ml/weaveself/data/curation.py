"""GPT_Curation_Node (Track B / Requirement 12).

Turns raw interactions into clean :class:`~weaveself.contracts.training_pair.TrainingPair`
rows. This is the *only* component in WeaveSelf permitted to call GPT (Req 12.2);
any GPT client import is kept lazy and local to :class:`GPTCurator` so a static
dependency check confirms no other module imports it.

Design contract:

* every emitted Training_Pair conforms to the Requirement 4 schema (Req 12.1,
  12.3 / Property 15);
* an interaction from which no valid Training_Pair can be produced is discarded
  and counted, so ``len(pairs) + discarded == len(interactions)`` and
  ``discarded`` equals the number of uncurable interactions (Req 12.4 /
  Property 16);
* the curation *model* is swappable: any object implementing :class:`Curator`
  (a GPT-backed one or a local model) produces the same schema (Req 12.3).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Protocol, Sequence, runtime_checkable

from weaveself.contracts.training_pair import (
    TrainingPair,
    validate_training_pair,
)


@runtime_checkable
class Curator(Protocol):
    """A swappable curation model.

    Implementations turn one raw interaction into a single Training_Pair, or
    return ``None`` when the interaction cannot be curated. The node validates
    every returned pair, so a curator that returns an ill-formed mapping is
    treated exactly like an uncurable interaction.
    """

    def curate(
        self, interaction: Mapping[str, object], unit_label: str
    ) -> Mapping[str, object] | TrainingPair | None: ...


@dataclass
class CurationResult:
    """Outcome of curating one Unit's interactions."""

    pairs: list[TrainingPair] = field(default_factory=list)
    discarded: int = 0

    @property
    def emitted(self) -> int:
        return len(self.pairs)


class GPTCurationNode:
    """The single GPT-calling curation node (Req 12).

    Holds a swappable :class:`Curator`. The default curator is local and never
    touches GPT; inject :class:`GPTCurator` to use GPT.
    """

    def __init__(self, curator: Curator | None = None) -> None:
        self._curator: Curator = curator or HeuristicLocalCurator()

    def curate_interactions(
        self,
        interactions: Sequence[Mapping[str, object]],
        unit_label: str,
    ) -> CurationResult:
        """Curate one Unit's raw interactions into Training_Pairs.

        Discards any interaction the curator cannot turn into a schema-valid
        Training_Pair and counts it, guaranteeing conservation (Property 16).
        """
        result = CurationResult()
        for interaction in interactions:
            pair = self._safe_curate(interaction, unit_label)
            if pair is None:
                result.discarded += 1
                continue
            result.pairs.append(pair)
        return result

    def _safe_curate(
        self, interaction: Mapping[str, object], unit_label: str
    ) -> TrainingPair | None:
        try:
            raw = self._curator.curate(interaction, unit_label)
        except Exception:
            # A curator that errors on an interaction is treated as unable to
            # produce a valid pair: discard and count it (Req 12.4).
            return None
        if raw is None:
            return None
        try:
            pair = validate_training_pair(raw)
        except (ValueError, TypeError):
            # Curator returned something non-conformant; discard (Property 15
            # holds because only validated pairs are ever emitted).
            return None
        # Force the Unit's label so the emitted pair always matches its Unit.
        if pair.unit_label != unit_label:
            pair = pair.model_copy(update={"unit_label": unit_label})
        return pair


class HeuristicLocalCurator:
    """A local, GPT-free curator usable as a Mock_Dependency and a real default.

    Emits a Training_Pair when the interaction carries usable prompt/completion
    text, otherwise returns ``None`` (uncurable). It accepts common shapes:

    * ``{"prompt": ..., "completion": ...}``
    * ``{"user": ..., "assistant": ...}``
    * ``{"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}``
    """

    def curate(
        self, interaction: Mapping[str, object], unit_label: str
    ) -> Mapping[str, object] | None:
        prompt, completion = self._extract(interaction)
        if not prompt or not completion:
            return None
        return {
            "prompt": prompt,
            "completion": completion,
            "unit_label": unit_label,
        }

    @staticmethod
    def _extract(interaction: Mapping[str, object]) -> tuple[str, str]:
        if not isinstance(interaction, Mapping):
            return "", ""
        prompt = interaction.get("prompt") or interaction.get("user") or ""
        completion = (
            interaction.get("completion") or interaction.get("assistant") or ""
        )
        if (not prompt or not completion) and "messages" in interaction:
            messages = interaction.get("messages") or []
            if isinstance(messages, Sequence):
                for msg in messages:
                    if not isinstance(msg, Mapping):
                        continue
                    role = msg.get("role")
                    content = msg.get("content") or ""
                    if role == "user" and not prompt:
                        prompt = content
                    elif role == "assistant" and not completion:
                        completion = content
        return str(prompt).strip(), str(completion).strip()


class GPTCurator:
    """A GPT-backed curator — the only place a GPT client is invoked (Req 12.2).

    The OpenAI client is imported lazily inside :meth:`curate` so that importing
    the rest of WeaveSelf never pulls in the GPT SDK; a static dependency check
    therefore finds the GPT import only here.
    """

    _SYSTEM_PROMPT = (
        "You convert one raw user interaction into a single clean training pair "
        "of the form {prompt, completion} that reflects the user's style and "
        "preferences. Reply with strict JSON and nothing else. If the "
        "interaction cannot be turned into a useful pair, reply with null."
    )

    def __init__(self, client: object | None = None, model: str = "gpt-4o-mini") -> None:
        self._client = client
        self._model = model

    def _get_client(self) -> object:
        if self._client is not None:
            return self._client
        # Lazy import keeps the GPT SDK out of every other module's import graph.
        from openai import OpenAI  # type: ignore

        self._client = OpenAI()
        return self._client

    def curate(
        self, interaction: Mapping[str, object], unit_label: str
    ) -> Mapping[str, object] | None:
        import json

        client = self._get_client()
        response = client.chat.completions.create(  # type: ignore[attr-defined]
            model=self._model,
            messages=[
                {"role": "system", "content": self._SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(dict(interaction))},
            ],
        )
        content = response.choices[0].message.content
        if content is None:
            return None
        content = content.strip()
        if not content or content.lower() == "null":
            return None
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            return None
        if not isinstance(parsed, Mapping):
            return None
        return {
            "prompt": parsed.get("prompt"),
            "completion": parsed.get("completion"),
            "unit_label": unit_label,
        }


class ResilientCurator:
    """Use a primary curator (e.g. GPT) when reachable, else a local fallback.

    Production behavior for the consolidation loop: prefer OpenAI curation, but
    if the GPT call raises (e.g. the API is unreachable / rate-limited), fall
    back to the local heuristic curator for that interaction so a network blip
    never discards the user's data. A primary result of ``None`` (the model
    judging an interaction uncurable) is respected and NOT overridden.
    """

    def __init__(self, primary: Curator, fallback: Curator | None = None) -> None:
        self._primary = primary
        self._fallback = fallback or HeuristicLocalCurator()

    def curate(
        self, interaction: Mapping[str, object], unit_label: str
    ) -> Mapping[str, object] | TrainingPair | None:
        try:
            return self._primary.curate(interaction, unit_label)
        except Exception:
            return self._fallback.curate(interaction, unit_label)
