"""Résolution de conflit (spec §8 "Résolution") : factual/temporal -> priorité
à la source la plus récente ; opinion -> synthèse multi-perspective."""

from typing import Literal

from pydantic import BaseModel

from src.reliability.conflict.types import ConflictReport


class ConflictResolution(BaseModel):
    strategy: Literal["none", "prefer_recent", "multi_perspective", "undetermined"]
    preferred_chunk_id: str | None = None
    reason: str | None = None


def resolve(report: ConflictReport, chunk_a: dict, chunk_b: dict) -> ConflictResolution:
    """Propose une stratégie de résolution pour un ConflictReport donné.

    `chunk_a`/`chunk_b` sont les mêmes dicts (chunk_id/payload) passés au
    détecteur — on y relit `valid_until`/`version_order` pour départager.
    """
    if not report.conflict:
        return ConflictResolution(strategy="none")

    if report.type in ("factual", "temporal"):
        preferred = _more_recent(chunk_a, chunk_b)
        if preferred is None:
            return ConflictResolution(
                strategy="undetermined",
                reason="Aucune des deux sources n'a de valid_until/version_order exploitable "
                "pour départager (voir status=None gap, docs/PHASE_3_SUMMARY.md §6).",
            )
        return ConflictResolution(
            strategy="prefer_recent",
            preferred_chunk_id=str(preferred.get("chunk_id")),
            reason="Source la plus récente selon valid_until/version_order.",
        )

    # opinion — ou type non déterminé : pas de tranchage possible, on garde les deux
    return ConflictResolution(
        strategy="multi_perspective",
        reason="Divergence d'interprétation plutôt qu'une contradiction factuelle tranchable — "
        "les deux perspectives doivent être présentées avec attribution de source.",
    )


def _more_recent(chunk_a: dict, chunk_b: dict) -> dict | None:
    """Retourne le chunk le plus récent selon valid_until puis version_order,
    ou None si aucun signal n'est exploitable sur les deux chunks."""
    payload_a = chunk_a.get("payload", {})
    payload_b = chunk_b.get("payload", {})

    valid_until_a, valid_until_b = payload_a.get("valid_until"), payload_b.get("valid_until")
    if valid_until_a and valid_until_b and valid_until_a != valid_until_b:
        return chunk_a if valid_until_a > valid_until_b else chunk_b

    order_a, order_b = payload_a.get("version_order"), payload_b.get("version_order")
    if order_a is not None and order_b is not None and order_a != order_b:
        return chunk_a if order_a > order_b else chunk_b

    return None
