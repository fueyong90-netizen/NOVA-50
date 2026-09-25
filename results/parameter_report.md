# Parameter Budget Report — configs/model_nova.yaml

| Bucket | Count |
|---|---|
| embedding | 4,096,000 |
| attention | 7,864,320 |
| ffn | 20,966,400 |
| normalization | 10,752 |
| output_head | 0 |
| other | 0 |
| **TOTAL (unique trainable tensors)** | **32,937,472** |
| Conservative total (tied head counted separately) | 37,033,472 |

> The output projection reuses the input embedding tensor (nn.Linear.weight = nn.Embedding.weight), so it introduces no additional trainable tensor in this implementation.

- Budget: 50,000,000
- Percentage used: 65.8749%
- Status: **PASS**
