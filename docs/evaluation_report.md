# Rapport d'évaluation TRADE

## Mesure provisoire — Phase 2 (Baseline A & B)

**Date :** 2026-07-23  
**Statut :** provisoire — à refaire formellement en Phase 5 avec les 50 questions annotées et RAGAS.

### Méthodologie

- **Corpus :** fixtures de test (`sample_openapi.yaml`, `sample_markdown.md`, `sample_incidents.json`)
- **Requêtes :** 5 questions artisanales (non annotées)
- **Baseline A :** dense-only (BGE-M3 dense, cosine, k=10)
- **Baseline B :** dense + sparse (SPLADE) + RRF client-side (k=60)

### Résultats préliminaires

Les résultats ci-dessous sont indicatifs et servent à valider l'infrastructure avant l'évaluation formelle :

| Configuration | Observations |
|---|---|
| Baseline A (dense-only) | Retrieval fonctionnel, signaux cohérents sur les fixtures. |
| Baseline B (+sparse+RRF) | Combinaison opérationnelle, RRF fusionne correctement les deux listes. |

**Conclusion provisoire :** l'infrastructure de retrieval est fonctionnelle. L'évaluation quantitative rigoureuse (Recall@k, MRR, nDCG, RAGAS) sera menée en Phase 5 sur un jeu de 50 questions annotées manuellement.

---

## Architecture d'évaluation (Phase 5)

| Étape | Configuration | Métriques |
|---|---|---|
| Baseline A | Dense-only, DOM hiérarchique, Contextual Retrieval | Recall@k, MRR, nDCG |
| Baseline B | A + Sparse + RRF client-side | Idem + latence |
| Baseline C | B + Reranker | Idem |
| Baseline D | C + Génération + Citations + Sufficiency | RAGAS F/AR/CP/CR + Hallucination Rate |
| Baseline E | D + Abstention | RAGAS + taux abstention correcte |
| Baseline F | E + Conflict Detection | RAGAS + taux faux conflits + taux conflits manqués |