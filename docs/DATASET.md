# DATASET.md

## ⚠️ Statut : corpus de démonstration (toy), pas le corpus final de compétition

Cet environnement d'exécution (conteneur sandbox) n'a **pas d'accès réseau
général** — uniquement aux registres de paquets (PyPI, npm, GitHub pour le
code source, etc.), pas à des sources de données brutes comme Common Crawl,
Wikipedia dumps, ou des dépôts de code pour du pré-entraînement. Il est donc
impossible d'exécuter ici un vrai téléchargement de corpus web-scale.

Pour permettre à **tout** le pipeline (nettoyage, dédup, split, tokenizer,
training, évaluation) d'être réellement exécuté et testé de bout en bout, un
corpus synthétique a été généré localement par
`data/scripts/prepare_toy_corpus.py`, à partir de gabarits paramétrés
(nombres, noms, opérations variables) — donc du texte réellement varié,
pas de simples copier-coller.

**Ce corpus ne doit PAS être utilisé pour l'entraînement final soumis au
challenge.** Pour la vraie soumission, remplacer `data/raw/*.jsonl` par de
vraies données publiques et documenter ci-dessous :

## Sources (à compléter avec les vraies données)
| Bucket | Source réelle prévue | Licence | Statut |
|---|---|---|---|
| general_text | UNKNOWN (ex: Wikipedia dump, licence CC BY-SA) | UNKNOWN | NON TÉLÉCHARGÉ |
| technical_text | UNKNOWN | UNKNOWN | NON TÉLÉCHARGÉ |
| mathematics | UNKNOWN | UNKNOWN | NON TÉLÉCHARGÉ |
| code | UNKNOWN (ex: The Stack, licences permissives filtrées) | UNKNOWN | NON TÉLÉCHARGÉ |
| reasoning | UNKNOWN | UNKNOWN | NON TÉLÉCHARGÉ |
| documentation | UNKNOWN | UNKNOWN | NON TÉLÉCHARGÉ |

## Corpus jouet actuellement présent dans le dépôt
Généré automatiquement — les statistiques ci-dessous sont calculées, pas
inventées (voir `data/raw/MANIFEST.json` et `data/processed/CLEAN_STATS.json`
pour les chiffres exacts au dernier run) :

- 6 catégories, gabarits paramétrés en anglais
- Volume brut : ~1900 exemples avant nettoyage
- Après nettoyage + déduplication exacte (hash SHA-256) : ~1400 exemples uniques
- Après split train/val/test (95/5/5, split déterministe par hash pour éviter
  toute fuite) : voir `data/processed/{train,val,test}.jsonl`
- Après tokenisation (vocab BPE 2000, entraîné sur ce corpus jouet) :
  ~19 500 tokens d'entraînement — **ordres de grandeur bien en-dessous** de ce
  qu'il faut pour un vrai pré-entraînement (qui se compte en dizaines/centaines
  de millions de tokens minimum pour un modèle de cette taille)

## Nettoyage appliqué
`data/scripts/clean_and_dedupe.py` :
- normalisation des espaces et sauts de ligne
- filtrage des textes trop courts (< 5 caractères)
- déduplication exacte par hash SHA-256 (pas de dédup approximative/MinHash
  implémentée dans cette démo — à ajouter pour un vrai corpus web-scale, où
  les quasi-doublons sont fréquents)

## Split train/val/test
`data/scripts/split_and_shard.py` : split déterministe basé sur le hash du
texte (pas sur l'ordre des lignes), pour que train/val/test restent stables
même si le pipeline est relancé, et pour limiter le risque de fuite si le
même texte apparaissait dans plusieurs fichiers sources.

## Limites connues
- Corpus jouet, taille très réduite, vocabulaire volontairement petit (2000)
- Pas de filtrage de qualité au-delà de la déduplication exacte
- Pas de détection de langue (tout est en anglais dans ce corpus jouet, alors
  que le brief est en français — voir note dans le README)
- Aucune contamination train/test testée au-delà du split par hash (voir
  `evaluation/contamination.py` pour la vérification automatique)
