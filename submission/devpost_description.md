# NOVA-50 — Devpost Description (brouillon)

## Inspiration
La question posée par le Track 01 (TECH) — combien de capacité de langage
peut-on extraire d'un modèle sous un budget strict de 50M paramètres — sans
pouvoir vérifier les règles précises du challenge en ligne au moment du
développement (voir `docs/COMPETITION_RULES.md`).

## What it does
NOVA-50 est un modèle de langage causal entraîné from scratch, avec une
architecture optimisée (RMSNorm, RoPE, GQA, SwiGLU, weight tying) comparée
systématiquement à un baseline Transformer standard à travers un pipeline
complet : données → tokenizer → entraînement → évaluation → démo web.

## How we built it
Voir `docs/TECHNICAL_REPORT.md` pour le détail complet. En résumé : une
implémentation PyTorch unique paramétrable sert à la fois le baseline et
NOVA-50, garantissant une comparaison à code égal ; un vérificateur de budget
de paramètres automatique empêche tout dépassement des 50M ; le pipeline est
couvert par 23 tests automatisés.

## Challenges we ran into
- Impossible de vérifier les règles officielles du challenge en ligne
  pendant le développement (marqué `UNKNOWN` partout où c'est pertinent)
- Pas d'accès à un vrai corpus web-scale ni à un GPU dans l'environnement de
  développement — le projet inclut donc un corpus jouet synthétique et des
  runs de démonstration miniatures, clairement labellisés comme tels

## Accomplishments that we're proud of
- Pipeline réellement fonctionnel de bout en bout (pas seulement une maquette)
- Vérification automatique et testée du budget de paramètres
- Ablations exécutées et documentées honnêtement, y compris quand les
  résultats sont faibles ou non concluants

## What we learned
[À compléter par l'équipe après les runs pleine échelle]

## What's next for NOVA-50
Entraînement pleine échelle sur un vrai corpus et GPU, intégration du
curriculum learning dynamique, et alignement sur les benchmarks officiels
dès qu'identifiés.
