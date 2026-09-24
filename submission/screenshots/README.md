# Screenshots à réaliser manuellement

1. Terminal montrant `python scripts/count_parameters.py --config configs/model_nova.yaml`
   (capture le tableau ASCII + STATUS: PASS)
2. `results/plots/loss_curves.png` ouvert en plein écran
3. Terminal montrant `python evaluation/report.py ...` avec sa sortie complète
4. Navigateur ouvert sur `web/index.html`, un prompt rempli, une réponse générée visible
5. Terminal montrant `pytest -q` avec "23 passed"

Faire ces captures APRÈS avoir relancé `scripts/reproduce.py` dans
l'environnement final de soumission (pas seulement dans ce conteneur de dev).
