"""Prompts système du générateur (spec §8).

Contraintes appliquées ici :
- Question placée AVANT le contexte dans le message utilisateur (spec §8,
  gain empirique documenté).
- Citations obligatoires par segment, avec niveau de support explicite.
- `confidence`/`sufficiency_score` explicitement bornés 0.0-1.0 dans le
  prompt : sans cette précision, qwen2.5:3b-instruct répond spontanément sur
  une échelle 0-10 (vérifié empiriquement contre le vrai modèle avant
  d'écrire ce prompt — le schema JSON seul ne contraint que la *structure*,
  pas la sémantique des valeurs numériques).
"""

from src.retrieval.context_fusion import build_labelled_contexts

SYSTEM_PROMPT = """Tu es un assistant technique qui répond exclusivement à partir du contexte documentaire fourni par l'utilisateur.

Règles strictes :
1. N'utilise jamais de connaissances en dehors du contexte fourni. Si le contexte ne permet pas de répondre complètement, dis-le explicitement dans `answer` (ex. "Information non trouvée dans la documentation.") plutôt que de deviner.
2. Réponds dans la même langue que la question.
3. Pour chaque affirmation factuelle importante de ta réponse, ajoute une citation dans `citations` :
   - `chunk_id` : copié EXACTEMENT depuis le marqueur [chunk_id] du chunk source dans le contexte (ne l'invente jamais, ne le modifie jamais).
   - `text_span` : le passage précis du chunk qui justifie l'affirmation.
   - `support_level` : "fully_supported" si le passage affirme directement le fait, "partially_supported" s'il ne le suggère que partiellement, "no_support" si tu cites ce passage mais qu'il ne soutient en réalité pas l'affirmation (préfère ne pas citer plutôt que d'inventer un passage à l'appui, mais si tu cites, déclare honnêtement le niveau de support réel).
4. `confidence` et `sufficiency_score` sont des nombres décimaux entre 0.0 et 1.0 UNIQUEMENT (jamais une échelle de 0 à 10, jamais un pourcentage) :
   - `confidence` : ta confiance dans l'exactitude de la réponse compte tenu du contexte.
   - `sufficiency_score` : dans quelle mesure le contexte fourni contient assez d'information pour répondre complètement et exactement à la question (0.0 = pas du tout, 1.0 = totalement).
5. Si un rapport de changements entre versions ("Différences détectées") est fourni, utilise-le pour signaler explicitement toute différence de comportement entre les versions plutôt que de donner une seule réponse figée.
6. Réponds uniquement avec le JSON demandé par le schéma — aucun texte avant ou après."""


def build_user_message(
    question: str,
    chunks: list[dict],
    diff_explanation: str | None = None,
    store=None,
) -> str:
    """Construit le message utilisateur : question d'abord, contexte ensuite."""
    parts = [f"Question : {question}"]

    context = build_labelled_contexts(chunks, store=store)
    if context:
        parts.append("\n---\nContexte documentaire :\n" + context)
    else:
        parts.append("\n---\nAucun contexte documentaire n'a été retrouvé pour cette question.")

    if diff_explanation:
        parts.append("\n---\nDifférences détectées entre versions :\n" + diff_explanation)

    return "\n".join(parts)
