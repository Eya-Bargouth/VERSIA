# VERSIA — RAG hybride pour documentation technique versionnée

Système de questions-réponses basé sur un LLM couplé à un RAG hybride (dense + sparse), 100 % local (Ollama/vLLM, aucune API payante), conçu pour de la documentation technique hétérogène et versionnée (specs API, guides, réglementation...).

Au-delà d'un RAG classique : citations systématiques vers la source exacte, détection explicite de l'absence d'information (abstention plutôt qu'hallucination), et détection des contradictions entre versions d'un même document.

## Architecture en un coup d'œil

- **Une seule base de données : Qdrant** — vecteurs + payload riche (hiérarchie, version, statut), pas de graphe séparé.
- **DOM unifié** : PDF, YAML (OpenAPI), Markdown, JSON sont tous normalisés en un même arbre de nœuds avant chunking.
- **Chunking hiérarchique + Contextual Retrieval** : chaque chunk garde son lien parent/enfant et un préfixe contextuel.
- **Retrieval hybride** : recherche dense (BGE-M3) + sparse, fusion RRF côté client, reranking (BGE-Reranker-v2-m3).
- **Fiabilité avant génération** : `SufficiencyChecker` (le contexte permet-il de répondre ?) → génération avec citations → `ConflictDetector` (structurel + textuel) → `AbstentionGate` (répond / répond avec réserve / s'abstient).
- **Sources auto-découvertes** : tout dossier ajouté sous `raw/` devient une source interrogeable, sans configuration à écrire.

Spec complète : [`docs/architecture_finale_TRADE_v2_1.md`](docs/architecture_finale_TRADE_v2_1.md).

## Démarrage rapide

```bash
# Configuration
cp src/config/.env.example .env

# Services (Qdrant + Ollama + API)
cd docker && docker compose up -d
docker exec -it trade-ollama ollama pull qwen2.5:3b-instruct

# Vérifier que tout tourne
curl http://localhost:8000/health

# Ingérer le corpus (raw/, découverte automatique des sources)
python scripts/run_ingestion.py --raw-dir raw/

# Poser une question
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "...", "top_k": 10}'
```

Interface web de vérification manuelle : http://localhost:8000/ui — dashboard Qdrant : http://localhost:6333/dashboard

Guide complet (prérequis CPU/GPU, Redis, tests, inspection de la collection Qdrant, dépannage) : [`docs/GUIDE_TEST_COMPLET.md`](docs/GUIDE_TEST_COMPLET.md).

## Tests

```bash
pytest tests/unit/ -v                    # rapide, dépendances lourdes mockées
pytest tests/integration/ -v -m slow     # pipeline réel, nécessite Qdrant (+ Ollama pour certains tests)
```

`tests/unit/` ne prouve que la logique isolée ; `tests/integration/` exerce les vrais modèles/services. Détail des marqueurs pytest et des commandes par phase dans le guide de test.

## Documentation

| Document | Contenu |
|---|---|
| [`docs/architecture_finale_TRADE_v2_1.md`](docs/architecture_finale_TRADE_v2_1.md) | Spécification complète |
| [`docs/GUIDE_TEST_COMPLET.md`](docs/GUIDE_TEST_COMPLET.md) | Guide de test et d'utilisation détaillé |

## Prérequis

- Docker & Docker Compose, Python 3.11+, Git
- GPU NVIDIA optionnel — le système fonctionne sur CPU (plus lent), bascule automatique si CUDA est indisponible
