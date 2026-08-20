#!/usr/bin/env python3
"""Rattrape le champ `parent_path` (voir src/dom/utils.py::derive_parent_path)
sur les chunks déjà indexés dans Qdrant, ingérés avant l'ajout de ce
mécanisme (expansion par fratrie, voir hybrid_retriever.py::
_fetch_missing_siblings). Aucune ré-ingestion, aucun recalcul d'embedding —
uniquement une écriture de payload dérivée du `hierarchy_path` déjà présent.

Usage : python scripts/backfill_parent_path.py
"""

import sys
from collections import defaultdict
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import structlog
from qdrant_client import QdrantClient

from src.config.settings import get_settings
from src.dom.utils import derive_parent_path
from src.embeddings.vector_store import QdrantStore

logger = structlog.get_logger(__name__)

SET_PAYLOAD_BATCH_SIZE = 500


def main():
    settings = get_settings()
    client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    store = QdrantStore(client=client, collection_name=settings.qdrant_collection_name)
    store.ensure_parent_path_index()

    # groups[parent_path_value] = [point_id, ...] — regroupe les points par
    # valeur calculée pour minimiser le nombre d'appels set_payload (les
    # frères d'une même liste source partagent la même valeur).
    groups: dict[str, list] = defaultdict(list)
    already_set = 0
    no_group = 0
    total = 0

    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=settings.qdrant_collection_name,
            limit=1000,
            offset=offset,
            with_payload=["hierarchy_path", "parent_path"],
        )
        if not points:
            break
        for point in points:
            total += 1
            hierarchy_path = point.payload.get("hierarchy_path", "")
            parent_path = derive_parent_path(hierarchy_path)
            if parent_path is None:
                no_group += 1
                continue
            if point.payload.get("parent_path") == parent_path:
                already_set += 1
                continue
            groups[parent_path].append(point.id)
        if offset is None:
            break

    print(f"Scanné {total} points — {no_group} hors groupe (pas de suffixe [N]), "
          f"{already_set} déjà à jour, {sum(len(v) for v in groups.values())} à mettre à jour "
          f"dans {len(groups)} groupes distincts.")

    updated = 0
    for parent_path, point_ids in groups.items():
        for i in range(0, len(point_ids), SET_PAYLOAD_BATCH_SIZE):
            batch = point_ids[i : i + SET_PAYLOAD_BATCH_SIZE]
            client.set_payload(
                collection_name=settings.qdrant_collection_name,
                payload={"parent_path": parent_path},
                points=batch,
            )
            updated += len(batch)
            if updated % 5000 < SET_PAYLOAD_BATCH_SIZE:
                print(f"  ... {updated} points mis à jour")

    print(f"Terminé — {updated} points mis à jour au total.")


if __name__ == "__main__":
    main()
