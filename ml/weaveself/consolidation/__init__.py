"""Nightly consolidation loop (the "sleep" job).

After ~24h of interaction, this loop takes a Unit's accumulated chats from
Redis, curates them into clean training pairs (OpenAI or a local curator),
trains/updates the Unit's NKT-Mirror weight adapter on the *cumulative* corpus
(so yesterday is not forgotten), and runs an **eval-gate**: the new adapter is
only promoted if it actually improves held-out perplexity over the previous
adapter without catastrophically forgetting prior days. Every run is observed in
Weave (consolidation score, forgetting score, gate deviation, curation yield,
promote/reject decision).
"""

from weaveself.consolidation.consolidate import (
    ConsolidationResult,
    consolidate_unit,
)

__all__ = ["ConsolidationResult", "consolidate_unit"]
