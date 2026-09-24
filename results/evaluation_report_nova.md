# Evaluation Report

- Checkpoint: `experiments/nova/toy_run/best.ckpt`
- Model config: `configs/model_nova_tiny_demo.yaml`

## Perplexity (test split)
```
Shard: data/processed/shards/test.npy
Tokens évalués: 1,152
Loss moyenne (cross-entropy): 4.6565
Perplexité: 105.2700
```

## Reasoning (arithmetic exact-match, maison — pas officiel)
```
Exact-match arithmétique sur 30 questions : 0/30 = 0.0%
NOTE: benchmark maison sur corpus jouet, PAS un benchmark officiel du challenge (UNKNOWN).
```

## Contamination check
```
train: 1252 hash uniques
val:   73 hash uniques
test:  74 hash uniques
Fuite train<->val:  0
Fuite train<->test: 0
Fuite val<->test:   0
STATUS: OK — aucune fuite détectée
```

## Official benchmarks
```
UNKNOWN: aucun benchmark officiel identifié — voir docs/COMPETITION_RULES.md
NOT RUN.
```
