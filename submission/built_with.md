# Built With
- Python 3.12
- PyTorch (CPU) — implémentation maison des couches (RMSNorm, RoPE, GQA,
  SwiGLU) et de la boucle d'entraînement, aucun framework d'entraînement
  externe (pas de HuggingFace Trainer)
- SentencePiece — tokenizer BPE entraîné from scratch sur le corpus du projet
- NumPy — stockage des tokens en shards mémoire-mappés
- Matplotlib — génération des graphiques à partir des logs réels
- pytest — 23 tests automatisés
- HTTP stdlib Python (`http.server`) — API d'inférence locale, sans dépendance
  supplémentaire type Flask/FastAPI
- HTML/CSS/JS vanilla — démo web statique
