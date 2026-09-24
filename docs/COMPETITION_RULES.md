# COMPETITION_RULES.md — Global Innovation Build Challenge V2, Track 01: TECH

**Statut de vérification : la page officielle Devpost du "Global Innovation
Build Challenge V2" n'a pas pu être localisée** via recherche web au moment
de la rédaction (24 septembre 2026). Aucune requête n'a renvoyé de page
Devpost, de site officiel, ni de communiqué correspondant précisément à ce
nom d'événement.

Une page tierce ("GIBC V2 EXP-020" sur Hugging Face) prétendant documenter un
modèle de 49,86M paramètres avec hashs SHA-256 et provenance a été trouvée,
mais **elle n'est pas utilisée comme source ici** : rien ne prouve qu'elle
émane des organisateurs du challenge (n'importe qui peut publier ce type de
page), et il n'est pas possible de vérifier son authenticité. La suivre
comme référence reviendrait à fabriquer des résultats sur une base non
vérifiée — exactement ce que ce document doit éviter.

En conséquence, **toutes les règles ci-dessous sont marquées UNKNOWN** sauf
la contrainte de budget de paramètres, qui est donnée explicitement dans le
brief fourni par l'utilisateur et traitée comme règle de travail confirmée.

| Règle | Source | Interprétation technique | Conséquence pour NOVA-50 |
|---|---|---|---|
| Budget max 50 000 000 paramètres entraînables | Brief utilisateur (confirmé) | Total de tous les tenseurs `requires_grad=True` : embeddings d'entrée, poids d'attention, FFN, normalisation, tête de sortie | `scripts/count_parameters.py` doit échouer (exit code ≠ 0) si dépassement |
| Définition exacte de "trainable" (buffers exclus ? weight tying compté une fois ?) | UNKNOWN | Hypothèse de travail : weight tying (embedding = output head) compte les paramètres partagés **une seule fois** dans le total, car ils ne forment qu'un seul tenseur physique en mémoire | À documenter comme hypothèse explicite dans le rapport final, à corriger si les règles réelles diffèrent |
| Modèles pré-entraînés externes autorisés ? | UNKNOWN | Le brief exige "from scratch" | NOVA-50 n'utilise aucun poids pré-entraîné, aucune distillation, aucun fine-tuning d'un modèle existant |
| Datasets autorisés / interdits | UNKNOWN | — | Seules des données publiques et librement licenciées sont utilisées ; sources et licences documentées dans `docs/DATASET.md` |
| Benchmarks obligatoires | UNKNOWN | — | Suite d'évaluation générique (perplexité, exact-match sur tâches simples de raisonnement/logique) construite en interne ; à remplacer par les benchmarks officiels dès qu'ils seront identifiés |
| Format de soumission / limite de taille du dépôt | UNKNOWN | — | `NOVA-50.zip` produit en phase finale, gros fichiers exclus |
| Exigences de reporting (rapport, vidéo, screenshots) | UNKNOWN | Le brief utilisateur donne des exigences précises (rapport technique, script vidéo, captures) | Ces exigences du brief sont suivies comme cahier des charges de facto |
| Critères de jugement / pondération | UNKNOWN | — | `submission/judging_alignment.md` restera générique tant que les critères réels ne sont pas connus |

## Action à prendre
Si tu as un lien direct vers la page Devpost officielle (ou un PDF/texte des
règles), transmets-le : ce fichier sera mis à jour avec les vraies règles et
les conséquences seront recalculées en conséquence (notamment la définition
exacte du budget de paramètres, qui peut changer le choix d'architecture).

En attendant, NOVA-50 est construit avec l'hypothèse la plus stricte et la
plus défendable : **50 000 000 paramètres entraînables au total, tout
inclus, weight tying compté une seule fois.**
