# TRADE — Guide de test et d'utilisation (pour l'encadrant)

> Guide pratique. Objectif ici : permettre à quelqu'un qui n'a jamais touché au projet de le lancer, l'ingérer, l'interroger, le tester et vérifier ce qu'il contient, de A à Z.
>
> Rédigé le 2026-09-01. 

---

## Sommaire

1. [Vue d'ensemble](#1-vue-densemble)
2. [Prérequis](#2-prérequis)
3. [Configuration](#3-configuration)
4. [Démarrer les services](#4-démarrer-les-services)
5. [Vérifier que tout tourne](#5-vérifier-que-tout-tourne)
6. [Ingérer le corpus](#6-ingérer-le-corpus)
7. [Tests automatisés](#7-tests-automatisés)
8. [Interroger le système via l'API](#8-interroger-le-système-via-lapi)
9. [Interroger le système via l'interface web](#9-interroger-le-système-via-linterface-web)
10. [Inspecter la collection Qdrant](#10-inspecter-la-collection-qdrant)
11. [Évaluation quantitative (optionnel, avancé)](#11-évaluation-quantitative-optionnel-avancé)
12. [Persistance des données et remise à zéro](#12-persistance-des-données-et-remise-à-zéro)
13. [Points d'attention connus](#13-points-dattention-connus)
14. [Checklist rapide](#14-checklist-rapide)

---

## 1. Vue d'ensemble

TRADE est un RAG hybride 100 % local : Qdrant (base vectorielle unique), Ollama (LLM local), BGE-M3 (embeddings) et BGE-Reranker (reranking), exposés via une API FastAPI et une petite interface web de vérification manuelle.

Quatre services entrent en jeu :

| Service | Rôle | Port |
|---|---|---|
| **Qdrant** | Base vectorielle (chunks + embeddings) | 6333 (HTTP), 6334 (gRPC) |
| **Ollama** | LLM local (génération, résumé contextuel optionnel) | 11434 |
| **API (FastAPI)** | Ingestion, requêtes, santé, UI | 8000 |
| **Redis** *(optionnel)* | Cache des réponses `/query` | 6379 |

---

## 2. Prérequis

- **Docker** + **Docker Compose** (v2, la commande `docker compose`, pas `docker-compose`)
- **Python 3.11+** si vous voulez lancer les scripts/tests en dehors du conteneur
- **GPU NVIDIA + drivers + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)** — optionnel, voir §13 pour ce qui l'utilise réellement aujourd'hui
- Espace disque : les modèles (BGE-M3, BGE-Reranker, poids Ollama) représentent plusieurs Go

### CPU seul (sans GPU)

Le système fonctionne entièrement sur CPU — plus lent (embeddings et génération), mais fonctionnel pour un test. `BGEEmbedder` (voir [`src/embeddings/bge_m3.py:47-51`](../src/embeddings/bge_m3.py)) détecte automatiquement l'absence de CUDA et bascule sur CPU avec un simple avertissement dans les logs (`cuda_unavailable_fallback_cpu`), sans erreur. Aucune configuration à changer pour tester sur une machine sans GPU.

---

## 3. Configuration

```bash
cp src/config/.env.example .env
```

Variables principales (toutes ont une valeur par défaut raisonnable, voir [`src/config/settings.py`](../src/config/settings.py)) :

| Variable | Défaut | Description |
|---|---|---|
| `QDRANT_HOST` / `QDRANT_PORT` | `localhost` / `6333` | Connexion Qdrant |
| `QDRANT_COLLECTION_NAME` | `trade_chunks` | Nom de la collection |
| `LLM_PROVIDER` | `ollama` | `ollama` ou `vllm` |
| `LLM_MODEL` | `qwen2.5:3b-instruct` | Modèle utilisé pour génération/résumé |
| `LLM_BASE_URL` | `http://localhost:11434` | Endpoint du provider LLM |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | Modèle d'embeddings |
| `RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | Modèle de reranking |
| `RAW_DATA_DIR` | `./raw` | Corpus source (lecture seule, sauf upload UI) |
| `REDIS_HOST` / `REDIS_PORT` | `localhost` / `6379` | Cache réponses (dégrade silencieusement si absent) |

**Important** : en local (hors Docker), utilisez `localhost` partout. En Docker, `docker-compose.yml` réécrit ces variables pour utiliser les noms de conteneurs (`trade-qdrant`, `trade-ollama`, `trade-redis`) — vous n'avez rien à changer manuellement pour ce cas.

---

## 4. Démarrer les services

### 4.1 Qdrant + Ollama + API (via Docker)

```bash
cd docker
docker compose up -d
```

Cela démarre `trade-qdrant`, `trade-ollama` et `trade-api` (build de l'image API — première fois plus longue, télécharge PyTorch CPU + dépendances). Le montage GPU est déclaré côté `ollama` et côté `api` dans [`docker/docker-compose.yml`](../docker/docker-compose.yml) — sans GPU disponible, `docker compose up` échoue sur ces deux services. **Si vous n'avez pas de GPU**, retirez ou commentez les blocs `deploy.resources.reservations.devices` des services `ollama` et `api` avant de lancer, ou lancez seulement Qdrant (§4.1bis) et faites tourner l'API/les scripts en local avec Python.

### 4.1bis Qdrant seul (le plus simple pour tester le stockage)

```bash
docker compose -f docker/docker-compose.yml up -d qdrant
```

### 4.2 Télécharger un modèle Ollama

```bash
docker exec -it trade-ollama ollama pull qwen2.5:3b-instruct
```

(Adapter le nom du modèle à `LLM_MODEL` dans votre `.env` si différent. Le `README.md` mentionne `mistral:7b` — c'est un modèle plus ancien, `qwen2.5:3b-instruct` est le défaut actuel du code.)

### 4.3 Redis (optionnel — cache des réponses)

Redis **n'est pas déclaré** dans `docker-compose.yml` (voir commentaire dans le fichier) — à démarrer séparément si vous voulez tester le cache :

```bash
docker run -d --name trade-redis --network docker_default -p 6379:6379 redis:latest
```

`docker_default` est le nom de réseau généré par Compose pour le dossier `docker/` — vérifiez avec `docker network ls` s'il ne correspond pas. Redis est **optionnel** : si absent ou injoignable, `src/api/cache.py` dégrade silencieusement vers une génération normale (pas d'erreur, juste pas de cache).

### 4.4 Lancer l'API en local (sans Docker, pour itérer plus vite)

```bash
pip install -e .
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Nécessite Qdrant et Ollama joignables (Docker ou installation locale) et les variables du `.env` pointant vers `localhost`.

---

## 5. Vérifier que tout tourne

```bash
docker ps
# Doit lister trade-qdrant, trade-ollama, trade-api (et trade-redis si lancé)

curl http://localhost:8000/health
```

Réponse attendue :
```json
{"status": "healthy", "services": {"qdrant": "ok", "llm": "ok"}, "version": "2.1.0"}
```
`"status": "degraded"` indique un service injoignable — le détail est dans `services`.

---

## 6. Ingérer le corpus

Le corpus est auto-découvert depuis `raw/` : chaque dossier contenant directement des fichiers = une source (`source_id` = nom du dossier), aucun manifeste à écrire.

### 6.1 Via le script CLI (le plus direct)

```bash
python scripts/run_ingestion.py --raw-dir raw/
```

Options utiles :
- `--source-id <nom>` : n'ingérer qu'une seule source (nom du dossier sous `raw/`)
- `--file-filter <sous-chaîne>` : ne traiter que les fichiers dont le nom contient cette sous-chaîne
- `--prefix-method deterministic|llm` : méthode de préfixe contextuel (`deterministic` par défaut, sans coût LLM ; `llm` plus lent, demande Ollama joignable)
- `--qdrant-url <url>` : override de l'URL Qdrant

Le script affiche un rapport (documents/chunks/erreurs par source) à la fin.

### 6.2 Via l'API (déclenche en tâche de fond)

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"raw_dir": "raw"}'
# → {"job_id": "...", "status": "queued", ...}

curl http://localhost:8000/ingest/<job_id>
# → {"job_id": "...", "status": "done", "report": {...}}
```

L'état des jobs est en mémoire côté API — perdu si l'API redémarre en cours d'ingestion (acceptable : il suffit de relancer).

### 6.3 Via l'interface web (upload d'un nouveau document)

Voir §9 — bouton d'upload qui écrit sous `raw/<source_id>/` puis relance l'ingestion restreinte à cette source.

### 6.4 Note sur le cache local des chunks (`data/cache/chunks/`)

Ce cache stocke le résultat du *parsing + chunking* (avant embedding), pas les vecteurs finaux. Il est distinct de ce que Qdrant contient et sert à éviter de refaire ce travail (potentiellement coûteux, surtout en `--prefix-method llm`) si vous relancez une ingestion sur un fichier déjà vu. Voir §13 pour la discussion complète (à qui il sert, quand il peut être supprimé sans risque).

---

## 7. Tests automatisés

### 7.1 Distinction essentielle

- **`tests/unit/`** : dépendances lourdes (Qdrant, BGE-M3, reranker, LLM) **mockées**. Rapides, ne nécessitent aucun service lancé. Un "PASSED" ici valide la logique, pas le comportement réel avec les vrais modèles/services.
- **`tests/integration/`** : visent le vrai pipeline. **Nécessitent Qdrant lancé**, et selon le test, Ollama et/ou les vrais modèles BGE-M3/reranker chargés en mémoire (donc plus lents, quelques secondes à quelques minutes).

### 7.2 Tests unitaires (rapides, aucun service requis)

```bash
pytest tests/unit/ -v
```

### 7.3 Tests d'intégration

Nécessitent Qdrant lancé (§4). Certains sont marqués `slow` et exclus par défaut (`pyproject.toml` : `addopts = "-m 'not slow'"`).

```bash
# Intégration rapide (retrieval sur la collection réelle déjà peuplée)
pytest tests/integration/test_retrieval_hybrid_phase3.py -v
# -> skip proprement si Qdrant n'est pas accessible ou la collection est vide

# Ingestion complète sur le vrai corpus, avec le vrai BGE-M3 (Qdrant en mémoire, isolé)
pytest tests/integration/test_real_pipeline_phase2.py -v -m slow

# Bout-en-bout réel : Qdrant persistant + BGE-M3 + reranker + Ollama, ensemble
# (nécessite la collection déjà peuplée ET Ollama lancé — sinon skip)
pytest tests/integration/test_end_to_end_real.py -v -m slow
```

### 7.4 Suite complète

```bash
pytest tests/unit/ -v                    # rapide, systématique
pytest tests/integration/ -v -m slow     # complet, nécessite les services + corpus ingéré
```

### 7.5 Filtrer par phase

Chaque phase du projet a son marker pytest (`phase1`...`phase6`, déclarés dans `pyproject.toml`) :

```bash
pytest tests/unit/ -v -m phase4   # ex. génération, citation, abstention, conflits de version
```

---

## 8. Interroger le système via l'API

### 8.1 Poser une question

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Quels sont les paramètres requis pour créer un ordre sur Binance ?", "top_k": 10}'
```

Réponse : objet `PipelineResult` (`src/pipeline.py`) — réponse générée, citations exactes vers les chunks source, éventuels conflits de version détectés, ou abstention explicite si le contexte est insuffisant.

### 8.2 Documentation interactive

FastAPI expose Swagger automatiquement :

```
http://localhost:8000/docs
```

Pratique pour explorer/tester chaque endpoint (`/query`, `/ingest`, `/ingest/upload`, `/ingest/{job_id}`, `/health`) sans écrire de `curl`.

---

## 9. Interroger le système via l'interface web

Une petite UI de vérification manuelle est servie à part de la spec (ajout hors périmètre v2.1, montée sous `/ui` pour ne jamais entrer en conflit avec les routes API) :

```
http://localhost:8000/ui
```

Elle permet, sans ligne de commande :
- de voir le statut de santé du système (pastille verte/rouge, `GET /health`) ;
- de poser une question et voir la réponse, les citations, les conflits détectés (`POST /query`) ;
- d'uploader un nouveau document dans une source (`POST /ingest/upload`) ;
- de lancer une ingestion sur `raw/` et suivre le job (`POST /ingest`, `GET /ingest/{job_id}`).

C'est la manière la plus simple de faire une démo à l'encadrant sans terminal.

---

## 10. Inspecter la collection Qdrant

### 10.1 Interface web intégrée à Qdrant

```
http://localhost:6333/dashboard
```

Permet de parcourir visuellement la collection `trade_chunks`, voir le nombre de points, les payloads, et faire une recherche manuelle.

### 10.2 Via l'API REST de Qdrant

```bash
curl http://localhost:6333/collections/trade_chunks
# -> "points_count": nombre de chunks ingérés
```

### 10.3 Via Python (inspection fine, ex. vérifier les métadonnées d'un chunk)

```python
from qdrant_client import QdrantClient

client = QdrantClient("localhost", port=6333)

info = client.get_collection("trade_chunks")
print(f"Total chunks: {info.points_count}")

points, _ = client.scroll("trade_chunks", limit=5)
for p in points:
    print(p.payload["source_id"], "-", p.payload["text"][:80])
    print("  status:", p.payload.get("status"), "| version:", p.payload.get("version_tag"))
```

### 10.4 Le cache Redis (si lancé, §4.3)

```bash
docker exec -it trade-redis redis-cli
> KEYS *
> GET <clé>
```

---

## 11. Évaluation quantitative (optionnel, avancé)

Le projet inclut des scripts d'évaluation utilisés pour produire les métriques du rapport de synthèse (`docs/rapport_validation_encadrant.md`, §7). Ils ne sont pas nécessaires pour un test fonctionnel simple, mais utiles pour reproduire une mesure :

```bash
# Génère des réponses sur un jeu de questions et sauvegarde les résultats bruts
python scripts/run_baseline_generation.py --questions-path data/eval/questions_sample_15_mixed.jsonl --out-prefix mon_run

# Calcule des métriques de qualité (citation accuracy, etc.) sur un run existant
python scripts/benchmark_quality_metrics.py
```

Nécessitent Qdrant peuplé + Ollama lancé. `scripts/run_eval.py` est un placeholder non implémenté — ignorez-le.

---

## 12. Persistance des données et remise à zéro

### 12.1 Qdrant est-il persistant ?

**Oui.** `docker-compose.yml` déclare Qdrant avec un volume Docker nommé :

```yaml
volumes:
  - qdrant_storage:/qdrant/storage
...
volumes:
  qdrant_storage:
```

Les données survivent à `docker compose stop` / `restart` / `down` (sans `-v`), et à un redémarrage complet de la machine. Elles ne sont perdues que si vous supprimez explicitement le volume :

```bash
docker compose down -v          # supprime aussi qdrant_storage et ollama_data — irréversible
# ou cibler uniquement Qdrant :
docker volume rm docker_qdrant_storage
```

### 12.2 Remise à zéro propre (repartir d'une collection vide)

```bash
docker compose down -v          # supprime volumes Qdrant + Ollama
docker compose up -d qdrant
python scripts/run_ingestion.py --raw-dir raw/
```

### 12.3 Le cache de chunks (`data/cache/chunks/`)

Peut être supprimé sans perdre de données côté Qdrant — il ne fait qu'accélérer une éventuelle ré-ingestion (voir §13 pour le détail).

---

## 13. Points d'attention connus

- **GPU côté conteneur `api`** : `docker-compose.yml` réserve un GPU pour le service `api`, mais [`docker/Dockerfile.api`](../docker/Dockerfile.api) installe explicitement la version **CPU** de PyTorch (`--index-url https://download.pytorch.org/whl/cpu`, avec une justification en commentaire liée à la compatibilité `torchvision`/`docling`). Résultat : même avec un GPU réservé, `torch.cuda.is_available()` renvoie `False` dans ce conteneur, et l'embedder bascule sur CPU (log `cuda_unavailable_fallback_cpu`). **Seul `ollama` utilise réellement le GPU aujourd'hui.** Pour faire tourner BGE-M3/reranker sur GPU, il faudrait soit lancer l'API en local (hors conteneur, avec un PyTorch CUDA installé), soit modifier `Dockerfile.api` pour installer un PyTorch CUDA — ce n'est pas fait pour l'instant.
- **Redis n'est pas dans `docker-compose.yml`** : à démarrer manuellement (§4.3) si vous voulez tester le cache de réponses. Son absence ne casse rien (dégradation silencieuse, voir `src/api/cache.py`).
- **`README.md` et `docs/TEST_QDRANT_PERSISTENT.md` contiennent des commandes obsolètes** (`--manifest-dir tests/fixtures/manifests`) : le système de manifeste a été supprimé (Phase 5, découverte automatique des sources depuis `raw/`). Utilisez `--raw-dir raw/` seul, comme indiqué dans ce guide.
- **État des jobs d'ingestion non persistant** : stocké en mémoire dans l'API (`app.state.jobs`), perdu si l'API redémarre en cours de job — relancer l'ingestion suffit, ce n'est pas un bug bloquant.
- **Cache de chunks (`data/cache/chunks/`) vs Qdrant** : ne stockent pas la même chose. Qdrant contient le produit final (chunks + vecteurs) servant aux requêtes ; le cache de chunks contient une étape intermédiaire (texte chunké, avant embedding), utile pour accélérer une ré-ingestion du même fichier (surtout en `--prefix-method llm`, qui appelle le LLM par chunk). Supprimer ce cache ne fait perdre aucune donnée interrogeable — au pire, la prochaine ingestion sera un peu plus lente sur les fichiers déjà vus.

---

## 14. Checklist rapide

```bash
# 1. Configuration
cp src/config/.env.example .env

# 2. Services
cd docker && docker compose up -d qdrant ollama   # + api si build Docker souhaité
docker exec -it trade-ollama ollama pull qwen2.5:3b-instruct

# 3. Santé
curl http://localhost:8000/health

# 4. Ingestion
python scripts/run_ingestion.py --raw-dir raw/

# 5. Vérifier Qdrant
curl http://localhost:6333/collections/trade_chunks
# ou http://localhost:6333/dashboard

# 6. Tests
pytest tests/unit/ -v
pytest tests/integration/ -v -m slow

# 7. Interroger
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" \
  -d '{"question": "...", "top_k": 10}'
# ou interface : http://localhost:8000/ui
```
