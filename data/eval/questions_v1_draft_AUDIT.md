# Audit du jeu de test Phase 5 — questions_v1_draft.jsonl

**Total : 50 questions.** Catégories LLM-assistées (multi_source, ambiguous) à auditer en priorité — marquées ⚠️ si le rationale du LLM lui-même trahit un appariement artificiel.

## Factuelles (15)

### 1. Quel(s) paramètre(s) sont requis pour GET /v1/apps/secrets ?

**Réponse attendue :** Paramètre(s) requis : scope (object)

**Sources attendues :** ['raw/specs-api/stripe/spec3-v2323.yaml']

*Générée par template (template_openapi)*

---

### 2. Quel(s) paramètre(s) sont requis pour GET /api/v3/klines ?

**Réponse attendue :** Paramètre(s) requis : interval (string)

**Sources attendues :** ['raw/specs-api/binance/spot_api.yaml']

*Générée par template (template_openapi)*

---

### 3. Quel(s) paramètre(s) sont requis pour GET /v1/apps/secrets/find ?

**Réponse attendue :** Paramètre(s) requis : name (string), scope (object)

**Sources attendues :** ['raw/specs-api/stripe/spec3-v2323.yaml']

*Générée par template (template_openapi)*

---

### 4. Quel(s) paramètre(s) sont requis pour GET /api/v3/uiKlines ?

**Réponse attendue :** Paramètre(s) requis : interval (string)

**Sources attendues :** ['raw/specs-api/binance/spot_api.yaml']

*Générée par template (template_openapi)*

---

### 5. Quel(s) paramètre(s) sont requis pour GET /v1/billing/credit_balance_summary ?

**Réponse attendue :** Paramètre(s) requis : filter (object)

**Sources attendues :** ['raw/specs-api/stripe/spec3-v2323.yaml']

*Générée par template (template_openapi)*

---

### 6. Quel(s) paramètre(s) sont requis pour POST /api/v3/order/cancelReplace ?

**Réponse attendue :** Paramètre(s) requis : cancelReplaceMode (string)

**Sources attendues :** ['raw/specs-api/binance/spot_api.yaml']

*Générée par template (template_openapi)*

---

### 7. Quel(s) paramètre(s) sont requis pour GET /v1/billing/meters/{id}/event_summaries ?

**Réponse attendue :** Paramètre(s) requis : customer (string), end_time (integer), start_time (integer)

**Sources attendues :** ['raw/specs-api/stripe/spec3-v2323.yaml']

*Générée par template (template_openapi)*

---

### 8. Quel(s) paramètre(s) sont requis pour POST /api/v3/orderList/oco ?

**Réponse attendue :** Paramètre(s) requis : aboveType (string), belowType (string)

**Sources attendues :** ['raw/specs-api/binance/spot_api.yaml']

*Générée par template (template_openapi)*

---

### 9. Quel(s) paramètre(s) sont requis pour GET /v1/charges/search ?

**Réponse attendue :** Paramètre(s) requis : query (string)

**Sources attendues :** ['raw/specs-api/stripe/spec3-v2323.yaml']

*Générée par template (template_openapi)*

---

### 10. Quel(s) paramètre(s) sont requis pour POST /sapi/v1/margin/borrow-repay ?

**Réponse attendue :** Paramètre(s) requis : isIsolated (string), type (string)

**Sources attendues :** ['raw/specs-api/binance/spot_api.yaml']

*Générée par template (template_openapi)*

---

### 11. Quel(s) paramètre(s) sont requis pour GET /v1/credit_notes/preview ?

**Réponse attendue :** Paramètre(s) requis : invoice (string)

**Sources attendues :** ['raw/specs-api/stripe/spec3-v2323.yaml']

*Générée par template (template_openapi)*

---

### 12. Quel(s) paramètre(s) sont requis pour GET /sapi/v1/margin/borrow-repay ?

**Réponse attendue :** Paramètre(s) requis : type (string)

**Sources attendues :** ['raw/specs-api/binance/spot_api.yaml']

*Générée par template (template_openapi)*

---

### 13. Quel(s) paramètre(s) sont requis pour GET /v1/credit_notes/preview/lines ?

**Réponse attendue :** Paramètre(s) requis : invoice (string)

**Sources attendues :** ['raw/specs-api/stripe/spec3-v2323.yaml']

*Générée par template (template_openapi)*

---

### 14. Quel(s) paramètre(s) sont requis pour POST /sapi/v1/margin/order ?

**Réponse attendue :** Paramètre(s) requis : autoRepayAtCancel (boolean)

**Sources attendues :** ['raw/specs-api/binance/spot_api.yaml']

*Générée par template (template_openapi)*

---

### 15. Quel(s) paramètre(s) sont requis pour GET /v1/customers/search ?

**Réponse attendue :** Paramètre(s) requis : query (string)

**Sources attendues :** ['raw/specs-api/stripe/spec3-v2323.yaml']

*Générée par template (template_openapi)*

---

## Conflit de versions (15)

### 1. Qu'est-ce qui a changé dans la description de l'endpoint « GET:/v1/balance/history|param:type » entre la version legacy et la version v2213 de l'API Stripe ?

**Réponse attendue :** Version legacy : - **type** (string, optional): Only returns transactions of the given type. One of: `tax_fund`, `adjustment`, `advance`, `advance_funding`, `anticipation_repayment`, `application_fee`, `application_fee_refund`, `charge`, `climate_order_purchase`, `climate_order_refund`, `connect_collection_transfer`, `contribution`, `inbound_transfer`, `inbound_transfer_reversal`, `issuing_authorization_hold`, `issuing_authorization_release`, `issuing_dispute`, `issuing_transaction`, `obligation...

**Sources attendues :** ['raw/specs-api/stripe/spec3-legacy.yaml', 'raw/specs-api/stripe/spec3-v2213.yaml']

*Générée par template (template_diff)*

---

### 2. Qu'est-ce qui a changé dans la description de l'endpoint « POST:/v1/terminal/refunds » entre la version legacy et la version v2213 de l'API Stripe ?

**Réponse attendue :** Version legacy : ### POST /v1/terminal/refunds
**Summary:** Create a refund using a Terminal-supported device.
**Description:** <p>Internal endpoint for terminal use to create a refund for a card_present or card charge.</p>

<p>You can optionally refund only part of a charge.</p>

Version v2213 : ### POST /v1/terminal/refunds
**Summary:** Create a refund using a Terminal-supported device.
**Description:** <p>Internal endpoint for terminal use to create a refund for a card_present charge.
This en...

**Sources attendues :** ['raw/specs-api/stripe/spec3-legacy.yaml', 'raw/specs-api/stripe/spec3-v2213.yaml']

*Générée par template (template_diff)*

---

### 3. Qu'est-ce qui a changé dans la description de l'endpoint « GET:/v1/payment_intents/search|param:query » entre la version legacy et la version v2213 de l'API Stripe ?

**Réponse attendue :** Version legacy : - **query** (string, required): The search query string. See [search query language](https://docs.stripe.com/search#search-query-language) and the list of supported [query fields for payment intents](https://docs.stripe.com/search#query-fields-for-paymentintents).

Version v2213 : - **query** (string, required): The search query string. See [search query language](https://docs.stripe.com/search#search-query-language) and the list of supported [query fields for payment intents](h...

**Sources attendues :** ['raw/specs-api/stripe/spec3-legacy.yaml', 'raw/specs-api/stripe/spec3-v2213.yaml']

*Générée par template (template_diff)*

---

### 4. Qu'est-ce qui a changé dans la description de l'endpoint « GET:/v1/balance/history|param:type » entre la version legacy et la version v2293 de l'API Stripe ?

**Réponse attendue :** Version legacy : - **type** (string, optional): Only returns transactions of the given type. One of: `tax_fund`, `adjustment`, `advance`, `advance_funding`, `anticipation_repayment`, `application_fee`, `application_fee_refund`, `charge`, `climate_order_purchase`, `climate_order_refund`, `connect_collection_transfer`, `contribution`, `inbound_transfer`, `inbound_transfer_reversal`, `issuing_authorization_hold`, `issuing_authorization_release`, `issuing_dispute`, `issuing_transaction`, `obligation...

**Sources attendues :** ['raw/specs-api/stripe/spec3-legacy.yaml', 'raw/specs-api/stripe/spec3-v2293.yaml']

*Générée par template (template_diff)*

---

### 5. Qu'est-ce qui a changé dans la description de l'endpoint « GET:/v1/payment_intents/search|param:query » entre la version legacy et la version v2293 de l'API Stripe ?

**Réponse attendue :** Version legacy : - **query** (string, required): The search query string. See [search query language](https://docs.stripe.com/search#search-query-language) and the list of supported [query fields for payment intents](https://docs.stripe.com/search#query-fields-for-paymentintents).

Version v2293 : - **query** (string, required): The search query string. See [search query language](https://docs.stripe.com/search#search-query-language) and the list of supported [query fields for payment intents](h...

**Sources attendues :** ['raw/specs-api/stripe/spec3-legacy.yaml', 'raw/specs-api/stripe/spec3-v2293.yaml']

*Générée par template (template_diff)*

---

### 6. Qu'est-ce qui a changé dans la description de l'endpoint « GET:/v1/balance_transactions|param:type » entre la version legacy et la version v2293 de l'API Stripe ?

**Réponse attendue :** Version legacy : - **type** (string, optional): Only returns transactions of the given type. One of: `tax_fund`, `adjustment`, `advance`, `advance_funding`, `anticipation_repayment`, `application_fee`, `application_fee_refund`, `charge`, `climate_order_purchase`, `climate_order_refund`, `connect_collection_transfer`, `contribution`, `inbound_transfer`, `inbound_transfer_reversal`, `issuing_authorization_hold`, `issuing_authorization_release`, `issuing_dispute`, `issuing_transaction`, `obligation...

**Sources attendues :** ['raw/specs-api/stripe/spec3-legacy.yaml', 'raw/specs-api/stripe/spec3-v2293.yaml']

*Générée par template (template_diff)*

---

### 7. Qu'est-ce qui a changé dans la description de l'endpoint « GET:/v1/payment_intents/search|param:query » entre la version legacy et la version v2323 de l'API Stripe ?

**Réponse attendue :** Version legacy : - **query** (string, required): The search query string. See [search query language](https://docs.stripe.com/search#search-query-language) and the list of supported [query fields for payment intents](https://docs.stripe.com/search#query-fields-for-paymentintents).

Version v2323 : - **query** (string, required): The search query string. See [search query language](https://docs.stripe.com/search#search-query-language) and the list of supported [query fields for payment intents](h...

**Sources attendues :** ['raw/specs-api/stripe/spec3-legacy.yaml', 'raw/specs-api/stripe/spec3-v2323.yaml']

*Générée par template (template_diff)*

---

### 8. Qu'est-ce qui a changé dans la description de l'endpoint « POST:/v1/credit_notes » entre la version legacy et la version v2323 de l'API Stripe ?

**Réponse attendue :** Version legacy : ### POST /v1/credit_notes
**Summary:** Create a credit note
**Description:** <p>Issue a credit note to adjust the amount of a finalized invoice. A credit note will first reduce the invoice’s <code>amount_remaining</code> (and <code>amount_due</code>), but not below zero.
This amount is indicated by the credit note’s <code>pre_payment_amount</code>. The excess amount is indicated by <code>post_payment_amount</code>, and it can result in any combination of the following:</p>

<ul>...

**Sources attendues :** ['raw/specs-api/stripe/spec3-legacy.yaml', 'raw/specs-api/stripe/spec3-v2323.yaml']

*Générée par template (template_diff)*

---

### 9. Qu'est-ce qui a changé dans la description de l'endpoint « GET:/v1/balance/history » entre la version legacy et la version v2323 de l'API Stripe ?

**Réponse attendue :** Version legacy : ### GET /v1/balance/history
**Summary:** List all balance transactions
**Description:** <p>Returns a list of transactions that have contributed to the Stripe account balance (for example, charges, transfers, and so on). The transactions return in sorted order, with the most recent transactions appearing first.</p>

<p>The previous name of this endpoint was “Balance history,” and it used the path <code>/v1/balance/history</code>.</p>

Version v2323 : ### GET /v1/balance/history
*...

**Sources attendues :** ['raw/specs-api/stripe/spec3-legacy.yaml', 'raw/specs-api/stripe/spec3-v2323.yaml']

*Générée par template (template_diff)*

---

### 10. Qu'est-ce qui a changé dans la description de l'endpoint « POST:/v1/payment_intents/{intent}/increment_authorization » entre la version v2213 et la version v2293 de l'API Stripe ?

**Réponse attendue :** Version v2213 : ### POST /v1/payment_intents/{intent}/increment_authorization
**Summary:** Increment an authorization
**Description:** <p>Perform an incremental authorization on an eligible
<a href="/docs/api/payment_intents/object">PaymentIntent</a>. To be eligible, the
PaymentIntent’s status must be <code>requires_capture</code> and
<a href="/docs/api/charges/object#charge_object-payment_method_details-card_present-incremental_authorization_supported">incremental_authorization_supported</a>
mu...

**Sources attendues :** ['raw/specs-api/stripe/spec3-v2213.yaml', 'raw/specs-api/stripe/spec3-v2293.yaml']

*Générée par template (template_diff)*

---

### 11. Qu'est-ce qui a changé dans la description de l'endpoint « DELETE:/v1/subscriptions/{subscription_exposed_id} » entre la version v2213 et la version v2293 de l'API Stripe ?

**Réponse attendue :** Version v2213 : ### DELETE /v1/subscriptions/{subscription_exposed_id}
**Summary:** Cancel a subscription
**Description:** <p>Cancels a customer’s subscription immediately. The customer won’t be charged again for the subscription. After it’s canceled, you can no longer update the subscription or its <a href="/metadata">metadata</a>.</p>

<p>Any pending invoice items that you’ve created are still charged at the end of the period, unless manually <a href="/api/invoiceitems/delete">deleted</a>. If ...

**Sources attendues :** ['raw/specs-api/stripe/spec3-v2213.yaml', 'raw/specs-api/stripe/spec3-v2293.yaml']

*Générée par template (template_diff)*

---

### 12. Qu'est-ce qui a changé dans la description de l'endpoint « POST:/v1/subscriptions/{subscription}/resume » entre la version v2213 et la version v2293 de l'API Stripe ?

**Réponse attendue :** Version v2213 : ### POST /v1/subscriptions/{subscription}/resume
**Summary:** Resume a subscription
**Description:** <p>Initiates resumption of a paused subscription, optionally resetting the billing cycle anchor and creating prorations. If no resumption invoice is generated, the subscription becomes <code>active</code> immediately. If a resumption invoice is generated, the subscription remains <code>paused</code> until the invoice is paid or marked uncollectible. If the invoice is not paid by t...

**Sources attendues :** ['raw/specs-api/stripe/spec3-v2213.yaml', 'raw/specs-api/stripe/spec3-v2293.yaml']

*Générée par template (template_diff)*

---

### 13. Qu'est-ce qui a changé dans la description de l'endpoint « POST:/v1/payment_intents/{intent}/increment_authorization » entre la version v2213 et la version v2323 de l'API Stripe ?

**Réponse attendue :** Version v2213 : ### POST /v1/payment_intents/{intent}/increment_authorization
**Summary:** Increment an authorization
**Description:** <p>Perform an incremental authorization on an eligible
<a href="/docs/api/payment_intents/object">PaymentIntent</a>. To be eligible, the
PaymentIntent’s status must be <code>requires_capture</code> and
<a href="/docs/api/charges/object#charge_object-payment_method_details-card_present-incremental_authorization_supported">incremental_authorization_supported</a>
mu...

**Sources attendues :** ['raw/specs-api/stripe/spec3-v2213.yaml', 'raw/specs-api/stripe/spec3-v2323.yaml']

*Générée par template (template_diff)*

---

### 14. Qu'est-ce qui a changé dans la description de l'endpoint « DELETE:/v1/subscriptions/{subscription_exposed_id} » entre la version v2213 et la version v2323 de l'API Stripe ?

**Réponse attendue :** Version v2213 : ### DELETE /v1/subscriptions/{subscription_exposed_id}
**Summary:** Cancel a subscription
**Description:** <p>Cancels a customer’s subscription immediately. The customer won’t be charged again for the subscription. After it’s canceled, you can no longer update the subscription or its <a href="/metadata">metadata</a>.</p>

<p>Any pending invoice items that you’ve created are still charged at the end of the period, unless manually <a href="/api/invoiceitems/delete">deleted</a>. If ...

**Sources attendues :** ['raw/specs-api/stripe/spec3-v2213.yaml', 'raw/specs-api/stripe/spec3-v2323.yaml']

*Générée par template (template_diff)*

---

### 15. Qu'est-ce qui a changé dans la description de l'endpoint « POST:/v1/subscriptions/{subscription}/resume » entre la version v2213 et la version v2323 de l'API Stripe ?

**Réponse attendue :** Version v2213 : ### POST /v1/subscriptions/{subscription}/resume
**Summary:** Resume a subscription
**Description:** <p>Initiates resumption of a paused subscription, optionally resetting the billing cycle anchor and creating prorations. If no resumption invoice is generated, the subscription becomes <code>active</code> immediately. If a resumption invoice is generated, the subscription remains <code>paused</code> until the invoice is paid or marked uncollectible. If the invoice is not paid by t...

**Sources attendues :** ['raw/specs-api/stripe/spec3-v2213.yaml', 'raw/specs-api/stripe/spec3-v2323.yaml']

*Générée par template (template_diff)*

---

## Abstention (11)

### 1. Quelle est la politique de remboursement pour un billet d'avion annulé ?

**Sources attendues :** (aucune — abstention attendue)

*Écrite manuellement — hors corpus car : aucun contenu voyage/billetterie dans le corpus*

---

### 2. Quels sont les frais de trading appliqués par Binance sur les paires USDT ?

**Sources attendues :** (aucune — abstention attendue)

*Écrite manuellement — hors corpus car : spot_api.yaml décrit les endpoints, pas la grille tarifaire*

---

### 3. Quel SLA de disponibilité Alpaca garantit-il contractuellement pour son API ?

**Sources attendues :** (aucune — abstention attendue)

*Écrite manuellement — hors corpus car : incidents.json liste des incidents passés, pas un engagement contractuel de SLA*

---

### 4. Comment activer l'authentification à deux facteurs sur le Dashboard Stripe ?

**Sources attendues :** (aucune — abstention attendue)

*Écrite manuellement — hors corpus car : guide d'usage du Dashboard, pas couvert par la spec OpenAPI ingérée*

---

### 5. Que recommande OWASP pour sécuriser un cluster Kubernetes ?

**Sources attendues :** (aucune — abstention attendue)

*Écrite manuellement — hors corpus car : seuls 3 cheat sheets OWASP sont ingérés (Authentication, Input Validation, REST Security) — pas Docker/Kubernetes*

---

### 6. Quelle est la limite de rate limiting de l'API REST de Coinbase ?

**Sources attendues :** (aucune — abstention attendue)

*Écrite manuellement — hors corpus car : Coinbase n'est pas une source du corpus (seul Binance y figure côté crypto)*

---

### 7. Comment intégrer Apple Pay avec Stripe Terminal ?

**Sources attendues :** (aucune — abstention attendue)

*Écrite manuellement — hors corpus car : guide d'intégration, pas couvert par la spec OpenAPI (référence d'endpoints, pas de tutoriels)*

---

### 8. Quel est le montant maximum autorisé pour un virement SEPA instantané en France ?

**Sources attendues :** (aucune — abstention attendue)

*Écrite manuellement — hors corpus car : réglementation bancaire française, hors périmètre du corpus (Stripe/Binance/OWASP/GDPR/Alpaca)*

---

### 9. Quelle version minimale de TLS est requise pour se connecter à l'API Binance Futures ?

**Sources attendues :** (aucune — abstention attendue)

*Écrite manuellement — hors corpus car : seul spot_api.yaml est ingéré, pas l'API Futures de Binance*

---

### 10. Quel délai un responsable de traitement a-t-il pour notifier une autorité de contrôle après la découverte d'une violation de données mineure sans risque pour les personnes ?

**Sources attendues :** (aucune — abstention attendue)

*Écrite manuellement — hors corpus car : formulation volontairement ambiguë/piégeuse sur un cas d'exemption GDPR précis — teste si le système invente une réponse assurée plutôt que de signaler l'incertitude*

---

### 11. Quel est le numéro de téléphone du support client Stripe pour les urgences de sécurité ?

**Sources attendues :** (aucune — abstention attendue)

*Écrite manuellement — hors corpus car : coordonnées de support, jamais présentes dans une spec OpenAPI*

---

## Multi-sources (5)

### 1. Quelles différences y-a-t-il entre la TLS Client Authentication et l'API de vérification d'identité proposée par Stripe, et dans quelles situations pourraient-elles être utilisées ensemble ?

**Sources attendues :** ['raw\\security\\owasp-cheatsheets\\Authentication_Cheat_Sheet.md', 'raw\\specs-api\\stripe\\spec3-legacy.yaml']

*Généré par LLM (thème : "authentification à l'API")*

> Rationale LLM : Cette question nécessite de comparer les deux extraits fournis. La réponse doit mentionner que TLS Client Authentication est une méthode de vérification basée sur le certificat client, tandis que l'API de vérification d'identité de Stripe est un processus distinct pour valider les documents d'identité des utilisateurs via une session en ligne. Les deux peuvent être utilisées ensemble dans des scénarios où une authentification plus sécurisée et approfondie est nécessaire, combinant la sécurité du certificat client avec l'examen manuel ou automatisé des documents d'identité par Stripe.

Extraits utilisés :
- `owasp_cheatsheets` (raw\security\owasp-cheatsheets\Authentication_Cheat_Sheet.md) : '#### TLS Client Authentication'
- `stripe_specs` (raw\specs-api\stripe\spec3-legacy.yaml) : 'document > paths  ### POST /v1/identity/verification_sessions **Summary:** Create a VerificationSession **Description:** <p>Creates a VerificationSession object.</p>  <p>After the VerificationSession '

---

### 2. Quelle est la différence entre l'input validation mentionnée dans le cheat sheet OWASP et la certification requise par la réglementation GDPR ? Comment ces deux concepts s'intersectent-ils dans le contexte de la vérification de signature d'un webhook ?

**Sources attendues :** ['raw\\security\\owasp-cheatsheets\\REST_Security_Cheat_Sheet.md', 'raw\\regulation\\gdpr-full.pdf']

*Généré par LLM (thème : "vérification de signature de webhook")*

> Rationale LLM : Cette question nécessite une compréhension des deux sources documentaires pour répondre correctement. L'utilisateur doit comprendre que l'input validation est un concept technique d'OWASP, tandis que la certification mentionnée dans le GDPR est un aspect réglementaire. La réponse devrait expliquer comment ces concepts peuvent interagir lors de la vérification de signature d'un webhook, en tenant compte des validations nécessaires et des exigences de conformité réglementaire.

Extraits utilisés :
- `owasp_cheatsheets` (raw\security\owasp-cheatsheets\REST_Security_Cheat_Sheet.md) : '## Input validation'
- `gdpr_regulation` (raw\regulation\gdpr-full.pdf) : 'document  Certification'

---

### 3. Quelles mesures recommande OWASP pour protéger contre les attaques par injection et comment cela peut-il être appliqué dans le contexte d'une API Stripe ?

**Sources attendues :** ['raw\\security\\owasp-cheatsheets\\Authentication_Cheat_Sheet.md', 'raw\\specs-api\\stripe\\spec3-v2293.yaml', 'raw\\status-pages\\alpaca\\incidents.json']

*Généré par LLM (thème : "protection contre les attaques par injection")*

> Rationale LLM : Cette question nécessite de comparer l'information fournie par OWASP sur la protection contre les attaques par injection avec les spécifications d'API Stripe, combinant ainsi les informations des deux sources pour répondre correctement.

Extraits utilisés :
- `owasp_cheatsheets` (raw\security\owasp-cheatsheets\Authentication_Cheat_Sheet.md) : '### Protect Against Automated Attacks'
- `stripe_specs` (raw\specs-api\stripe\spec3-v2293.yaml) : ''
- `alpaca_incidents` (raw\status-pages\alpaca\incidents.json) : ''

---

### 4. Quelles différences y-a-t-il entre les recommandations pour la gestion des erreurs dans un service REST selon OWASP et les exigences concernant le code de conduite dans la réglementation GDPR ? Comment ces deux documents traitent-ils respectivement l'aspect des codes de réponse aux erreurs ?

**Sources attendues :** ['raw\\security\\owasp-cheatsheets\\REST_Security_Cheat_Sheet.md', 'raw\\regulation\\gdpr-full.pdf']

*Généré par LLM (thème : "gestion des erreurs et codes de réponse" ⚠️ **rationale suspect — lire avant de garder**)*

> Rationale LLM : Cette question nécessite d'examiner à la fois le document OWASP sur la gestion des erreurs en REST, qui devrait fournir des recommandations pratiques pour gérer les erreurs dans un service REST, et le document GDPR qui ne contient pas directement de renseignements sur l'erreur handling mais pourrait être utilisé pour comprendre comment traiter les informations sensibles concernant la protection des données personnelles. Cela nécessite une compréhension croisée des deux documents pour répondre correctement.

Extraits utilisés :
- `owasp_cheatsheets` (raw\security\owasp-cheatsheets\REST_Security_Cheat_Sheet.md) : '## Error handling'
- `gdpr_regulation` (raw\regulation\gdpr-full.pdf) : 'document  Codes of conduct'

---

### 5. Quelles sont les recommandations pour la gestion des incidents et l'erreur handling dans un service REST selon les sources fournies ? Comparez les mesures prises par Alpaca Incidents avec celles suggérées par OWASP pour le traitement des erreurs en RESTful services. Comment ces deux approches peuvent-elles être combinées pour améliorer la gestion des interruptions de service ?

**Sources attendues :** ['raw\\status-pages\\alpaca\\incidents.json', 'raw\\security\\owasp-cheatsheets\\REST_Security_Cheat_Sheet.md']

*Généré par LLM (thème : "gestion des incidents et interruptions de service")*

> Rationale LLM : Cette question nécessite d'analyser les informations fournies dans les deux sources documentaires, en comparant les pratiques recommandées par OWASP concernant l'erreur handling et celles mises en œuvre par Alpaca Incidents pour la gestion des incidents. Elle demande également une réflexion sur la manière de combiner ces approches pour améliorer la gestion des interruptions de service, ce qui nécessite un croisement d'informations provenant de deux sources différentes.

Extraits utilisés :
- `alpaca_incidents` (raw\status-pages\alpaca\incidents.json) : '## incidents'
- `owasp_cheatsheets` (raw\security\owasp-cheatsheets\REST_Security_Cheat_Sheet.md) : '## Error handling'

---

## Ambiguës (4)

### 1. Quelle est la différence entre les stratégies d'input validation mentionnées dans le document OWASP et la procédure de pagination des résultats décrite dans le règlement GDPR ?

**Sources attendues :** ['raw\\regulation\\gdpr-full.pdf', 'raw\\security\\owasp-cheatsheets\\Input_Validation_Cheat_Sheet.md']

*Généré par LLM (thème : "politique de pagination des résultats" ⚠️ **rationale suspect — lire avant de garder**)*

> Rationale LLM : Cette question nécessite de comparer les deux sources pour identifier les différences entre les stratégies d'input validation (OWASP) et la procédure de pagination des résultats (GDPR). Une réponse correcte devrait mentionner que le document OWASP parle de techniques d'input validation, tandis que le règlement GDPR ne décrit pas explicitement une procédure de pagination mais plutôt des principes de protection des données personnelles.

Extraits utilisés :
- `gdpr_regulation` (raw\regulation\gdpr-full.pdf) : 'document  Procedure'
- `owasp_cheatsheets` (raw\security\owasp-cheatsheets\Input_Validation_Cheat_Sheet.md) : '## Input Validation Strategies'

---

### 2. Quel est le maximum d'intervalle temporel autorisé entre les dates de début et fin pour récupérer l'historique des dépôts NFT sur la plateforme Binance, et quel est le régime de conservation des données selon la réglementation GDPR ?

**Sources attendues :** ['raw\\specs-api\\binance\\spot_api.yaml', 'raw\\regulation\\gdpr-full.pdf']

*Généré par LLM (thème : "durée de conservation des données" ⚠️ **rationale suspect — lire avant de garder**)*

> Rationale LLM : Cette question nécessite de comparer les informations fournies dans les deux sources documentaires. L'extrait de Binance indique que l'intervalle maximal entre startTime et endTime pour récupérer l'historique des dépôts NFT est de 90 jours, tandis que l'extrait de la réglementation GDPR ne fournit pas d'informations précises sur le régime de conservation des données, mais suggère qu'il existe un cadre légal à respecter concernant la confidentialité et la durée de conservation des données personnelles.

Extraits utilisés :
- `binance_spot` (raw\specs-api\binance\spot_api.yaml) : 'document > paths  ### GET /sapi/v1/nft/history/deposit **Summary:** Get NFT Deposit History(USER_DATA) **Description:** - The max interval between startTime and endTime is 90 days. - If startTime and '
- `gdpr_regulation` (raw\regulation\gdpr-full.pdf) : 'document  Confidentiality'

---

### 3. Quelles sont les recommandations pour la gestion des erreurs dans le format de réponse lorsqu'on répond à un défi de fraude dans l'API Stripe ?

**Sources attendues :** ['raw\\security\\owasp-cheatsheets\\REST_Security_Cheat_Sheet.md', 'raw\\specs-api\\stripe\\spec3-legacy.yaml']

*Généré par LLM (thème : "format de réponse en cas d'erreur")*

> Rationale LLM : Cette question nécessite d'analyser les deux extraits fournis. L'extrait du cheet sheet OWASP REST Security explique comment gérer les erreurs en général, tandis que celui de l'API Stripe spécifie la façon dont le service doit être utilisé pour répondre à un défi de fraude. Pour répondre correctement, il faut comprendre et combiner ces informations.

Extraits utilisés :
- `owasp_cheatsheets` (raw\security\owasp-cheatsheets\REST_Security_Cheat_Sheet.md) : '## Error handling'
- `stripe_specs` (raw\specs-api\stripe\spec3-legacy.yaml) : 'document > paths  ### POST /v1/test_helpers/issuing/authorizations/{authorization}/fraud_challenges/respond **Summary:** Respond to fraud challenge **Description:** <p>Respond to a fraud challenge on '

---

### 4. Quelles sont les recommandations pour la gestion des clés d'API et comment une expiraison peut-elle être gérée selon ces sources ?

**Sources attendues :** ['raw\\security\\owasp-cheatsheets\\REST_Security_Cheat_Sheet.md', 'raw\\status-pages\\alpaca\\incidents.json', 'raw\\specs-api\\stripe\\spec3-v2293.yaml']

*Généré par LLM (thème : "expiration et rotation des clés d'API" ⚠️ **rationale suspect — lire avant de garder**)*

> Rationale LLM : Cette question nécessite de comparer l'information sur la gestion des clés d'API trouvée dans le document OWASP avec celle concernant l'expiration des codes promotionnels dans le document Stripe. Il n'existe pas une seule source qui contient toutes les informations requises pour répondre correctement à cette question.

Extraits utilisés :
- `owasp_cheatsheets` (raw\security\owasp-cheatsheets\REST_Security_Cheat_Sheet.md) : '## API Keys'
- `alpaca_incidents` (raw\status-pages\alpaca\incidents.json) : 'document > incidents  ```json {   "id": "nslrlg6r2pk8",   "name": "Service degradation across multiple API\'s",   "status": "resolved",   "created_at": "2026-03-31T09:25:54.865-04:00",   "updated_at": '
- `stripe_specs` (raw\specs-api\stripe\spec3-v2293.yaml) : 'document > paths  ### POST /v1/promotion_codes **Summary:** Create a promotion code **Description:** <p>A promotion code points to an underlying promotion. You can optionally restrict the code to a sp'

---
