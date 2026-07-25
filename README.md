# TRADE — Hybrid RAG for Versioned Technical Documentation (v2.1)

## Phase 1 — Fondations

### Prérequis

- Docker & Docker Compose
- Python 3.11+
- Git

### Installation

```bash
# Cloner le repo
cd trade-rag

# Installer en mode éditable
pip install -e .

# Copier la configuration
cp src/config/.env.example .env
# Éditer .env selon ton environnement
```

### Lancer les services (Docker)

```bash
cd docker
docker compose up -d
```

Services démarrés :
- **Qdrant** : http://localhost:6333
- **Ollama** : http://localhost:11434
- **API FastAPI** : http://localhost:8000

### Télécharger un modèle Ollama

```bash
docker exec -it trade-ollama ollama pull mistral:7b
```

### Tests Phase 1

```bash
pytest tests/unit tests/integration/test_phase1.py -v -m phase1
```

### Vérification santé

```bash
curl http://localhost:8000/health
```
## Phase 2 — Ingestion & Métadonnées

### Ingestion progressive (une source à la fois)

```bash
# Tout le corpus
python scripts/run_ingestion.py --manifest-dir tests/fixtures/manifests --raw-dir raw/

# Une source spécifique
python scripts/run_ingestion.py --manifest-dir tests/fixtures/manifests --raw-dir raw/ --source-id stripe_specs