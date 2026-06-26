# Multi-Task Review Triage — Evaluation & Error Analysis

The evaluation and error-analysis workstream I owned on a three-person M.Eng. AI/ML capstone
(George Washington University, 2026). The team built a multi-task review-triage model: a BERT
encoder with five auxiliary heads (sentiment, emotion, topic, intent, urgency) feeding a four-class
triage classifier, trained on 12,000 Uber app-store reviews. The architecture and pseudo-labeling
were teammates' work. My job was measuring it: per-class metrics, the confusion matrix, error
analysis, and validating label quality.

## What's in the notebook

- **Section 4 — model evaluation.** The trained model's real confusion matrix (transcribed from
  Figure 4.6 of the written report), with per-class precision/recall/F1, macro/weighted F1, and an
  integrity check that recomputes the headline metrics from the matrix and confirms they reproduce
  the report's stated accuracy 0.8225, macro F1 0.7028, and weighted F1 0.8260 on the held-out
  1,200-review validation set.
- **Section 5 — model error analysis.** Adjacency of the model's misclassifications (97.2% are a
  single tier apart, computed from the confusion matrix) and a TF-IDF centroid-similarity view of
  why adjacent tiers blur.
- **Section 6 — label-quality validation.** The dataset shipped two independent label pipelines
  (a weak-supervision heuristic and a Claude Haiku pass) for the same reviews; treating them as two
  annotators audits the training labels. Neither is human-verified ground truth, so this measures
  label-pipeline consistency, not accuracy.
- **Appendix A** reports the model-variant sweeps (loss strategy, encoder, feature fusion, auxiliary
  heads) from the project's A100 runs, sourced to the written report.

## Headline findings

- **Final model:** 82.3% accuracy, 0.826 weighted F1, 0.703 macro F1 — verified directly from its
  confusion matrix. "No triage" is near-perfect (F1 0.95); the middle tiers carry the error.
- **Safe failure mode:** 97.2% of misclassifications are adjacent-tier; the model essentially never
  confuses "No triage" with a high-priority review.
- **Confusable classes are semantically confusable:** "Urgent" and "Immediate" review text sits at
  0.87 TF-IDF centroid similarity; "No triage" is cleanly separated (~0.13).
- **Label-quality caveat:** the two labeling pipelines agree only moderately (Cohen's κ ≈ 0.33),
  with the heuristic systematically over-escalating — a real bound on how far the triage numbers can
  be trusted.

## Run it

```bash
pip install pandas numpy scikit-learn matplotlib jupyter
jupyter notebook Triage_Evaluation_Error_Analysis.ipynb
```

Expected data files in `./data/`:

```
data/Triage_Dataset_unverified_v2.csv      # weak-supervision pipeline (12,000 rows)
data/haiku_1_safety_critical.csv           # Claude Haiku pass, by category
data/haiku_2_service_issues.csv
data/haiku_3_neutral_feedback.csv
data/haiku_4_positive_reviews.csv
```

`Triage_Evaluation_Error_Analysis.html` is a pre-rendered copy with all outputs and figures baked
in, viewable without Jupyter. To extend the error analysis to the model's actual per-review misses
(Figure 4.10's length-vs-error plot), drop the model's saved validation predictions into `data/`.
