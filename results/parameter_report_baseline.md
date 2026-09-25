# Parameter Budget Report — configs/model_baseline.yaml

| Bucket | Count |
|---|---|
| embedding | 2,560,000 |
| attention | 3,276,800 |
| ffn | 6,566,400 |
| normalization | 10,880 |
| output_head | 2,560,000 |
| other | 0 |
| **TOTAL (unique trainable tensors)** | **14,974,080** |
| Conservative total (tied head counted separately) | 14,974,080 |

> Weight tying is disabled for this config: embedding and output head are separate tensors, already counted separately above.

- Budget: 50,000,000
- Percentage used: 29.9482%
- Status: **PASS**
