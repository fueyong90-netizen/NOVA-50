#!/usr/bin/env python3
"""
Prépare un petit corpus de démonstration (PAS le corpus final d'entraînement).

Ce conteneur n'a pas d'accès réseau général (seulement les registres de
paquets : pypi, npm, github...), donc le téléchargement d'un vrai corpus
web-scale (ex: subset de C4 / The Pile / Wikipedia) ne peut pas être exécuté
ici. Ce script génère à la place un petit corpus synthétique avec des
gabarits paramétrés (nombres, noms, opérations variables) pour produire un
grand nombre d'exemples RÉELLEMENT UNIQUES, afin que le pipeline complet
(dédup, tokenizer, dataset, training, eval — y compris l'évaluation
arithmétique exacte) soit réellement exécutable de bout en bout, avec des
données honnêtement labellisées comme synthétiques.

Pour la vraie soumission, remplacer ce script par un vrai téléchargeur
(ex: Common Crawl / Wikipedia dump / GitHub code) exécuté avec accès réseau
complet, en respectant les licences (voir docs/DATASET.md).
"""
import argparse
import json
import random
from pathlib import Path

random.seed(42)

NAMES = ["John", "Maria", "Aiko", "Kwame", "Fatou", "Lucas", "Ines", "Yusuf", "Chidi", "Elena"]
FRUITS = ["apples", "oranges", "mangoes", "bananas", "pears", "lemons"]
GENERAL_TEMPLATES = [
    "The old lighthouse keeper watched the storm roll in from the {direction}.",
    "{name} poured two cups of tea and sat by the window, waiting for the rain to stop.",
    "Every morning the market in {place} fills with the smell of fresh bread and roasted coffee.",
    "The children ran across the field, chasing a kite that had slipped from {name}'s hand.",
    "History remembers few rulers as kindly as it remembers good bridge builders in {place}.",
    "A quiet river cuts through the valley near {place}, feeding fields farmed for centuries.",
    "The museum's oldest exhibit is a clay tablet no larger than a human palm, found near {place}.",
    "Long after the concert ended, the melody kept turning over in {name}'s mind.",
]
DIRECTIONS = ["west", "north", "east", "south", "coast", "mountains"]
PLACES = ["Cotonou", "Douala", "Lagos", "Accra", "Dakar", "Yaounde", "Porto-Novo", "Bamako"]

TECHNICAL_TEMPLATES = [
    "A transformer processes a sequence of length {n} by attending to every other position at once.",
    "Gradient descent with a learning rate of {lr} updates parameters to reduce the loss function.",
    "A hash table with {n} buckets maps keys to values using a distribution function.",
    "TCP guarantees ordered delivery of packets by acknowledging each of the {n} segments received.",
    "A binary search tree with {n} elements keeps lookups at logarithmic time complexity.",
    "Caching the result of a function call avoids repeating {n} milliseconds of computation.",
]

MATH_TEMPLATES = [
    ("What is {a} plus {b}?", "sum"),
    ("What is {a} minus {b}?", "diff"),
    ("What is {a} times {b}?", "prod"),
    ("The sum of {a} and {b} is", "sum"),
]

CODE_TEMPLATES = [
    "def add_{n}(a, b):\n    return a + b + {n}",
    "for i in range({n}):\n    print(i * i)",
    "class Counter{n}:\n    def __init__(self):\n        self.value = {n}",
    "SELECT name FROM users WHERE age > {n} ORDER BY name ASC;",
    "function multiplyBy{n}(x) {{\n  return x * {n};\n}}",
]

REASONING_TEMPLATES_QA = [
    ("{name} has {a} {fruit} and buys {b} more. How many {fruit} does {name} have now?", "sum"),
    ("{name} had {a} {fruit}, gave away {b}, and now has how many left?", "diff"),
]
REASONING_STATEMENTS = [
    "If all cats are mammals and all mammals are animals, then all cats are animals.",
    "Because the ground was wet and there were no clouds, the water likely came from a sprinkler.",
    "If no fish can fly, and a salmon is a fish, then a salmon cannot fly.",
]

DOC_TEMPLATES = [
    "## Installation\n\nRun `pip install -r requirements.txt` to install dependency set {n}.",
    "## Usage\n\nCall `train(config_{n})` with a valid configuration object to start training.",
    "### Parameters\n\n- `lr`: learning rate for run {n}.\n- `batch_size`: number of examples per step.",
]


def gen_general(n):
    return [
        random.choice(GENERAL_TEMPLATES).format(
            direction=random.choice(DIRECTIONS), name=random.choice(NAMES), place=random.choice(PLACES)
        )
        for _ in range(n)
    ]


def gen_technical(n):
    return [
        random.choice(TECHNICAL_TEMPLATES).format(
            n=random.randint(2, 500), lr=round(random.uniform(1e-4, 1e-2), 5)
        )
        for _ in range(n)
    ]


def gen_math(n):
    out = []
    for _ in range(n):
        q_t, kind = random.choice(MATH_TEMPLATES)
        a, b = random.randint(1, 50), random.randint(1, 50)
        q = q_t.format(a=a, b=b)
        ans = {"sum": a + b, "diff": a - b, "prod": a * b}[kind]
        out.append(f"Question: {q} Answer: {ans}.")
    return out


def gen_code(n):
    return [random.choice(CODE_TEMPLATES).format(n=random.randint(0, 999)) for _ in range(n)]


def gen_reasoning(n):
    out = []
    for _ in range(n):
        if random.random() < 0.5:
            out.append(random.choice(REASONING_STATEMENTS))
        else:
            q_t, kind = random.choice(REASONING_TEMPLATES_QA)
            a, b = random.randint(2, 20), random.randint(1, 10)
            name, fruit = random.choice(NAMES), random.choice(FRUITS)
            q = q_t.format(name=name, a=a, b=b, fruit=fruit)
            ans = a + b if kind == "sum" else max(a - b, 0)
            out.append(f"Question: {q} Answer: {ans}.")
    return out


def gen_documentation(n):
    return [random.choice(DOC_TEMPLATES).format(n=random.randint(1, 99)) for _ in range(n)]


BUCKETS = {
    "general_text": (gen_general, 400),
    "technical_text": (gen_technical, 350),
    "mathematics": (gen_math, 400),
    "code": (gen_code, 300),
    "reasoning": (gen_reasoning, 350),
    "documentation": (gen_documentation, 200),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data/raw")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for bucket, (fn, n) in BUCKETS.items():
        lines = fn(n)
        bucket_path = out_dir / f"{bucket}.jsonl"
        with open(bucket_path, "w") as f:
            for line in lines:
                f.write(json.dumps({
                    "text": line, "source": f"toy_corpus_synthetic/{bucket}",
                    "license": "synthetic, generated locally (demo data, not for final scoring)",
                }) + "\n")
        manifest.append({"bucket": bucket, "file": str(bucket_path), "n_examples": len(lines)})
        print(f"[{bucket}] {len(lines)} exemples -> {bucket_path}")

    with open(out_dir / "MANIFEST.json", "w") as f:
        json.dump({
            "note": "Corpus jouet synthétique, gabarits paramétrés — voir docs/DATASET.md",
            "buckets": manifest,
        }, f, indent=2)


if __name__ == "__main__":
    main()
