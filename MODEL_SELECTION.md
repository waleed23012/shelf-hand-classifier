# Model selection: shelf hand classifier (Lab 3)

## Setup
- Features: frozen DINOv2-base embeddings (768-dim), cached in data/embeddings.npz
- Train: 699 images, 10 videos. Test: 68 images, 4 videos never seen in training
- Leakage fix: 4 videos were in both train and test; their 40 test frames were moved to train
- Data slice = test video (each one is a different recording session)

## Candidates
| Candidate | F1 macro | Worst video F1 | AUC | p95 latency (head) | Size |
|---|---|---|---|---|---|
| logreg (C=0.1) | 0.696 | 0.533 | 0.721 | 0.15 ms | 7 KB |
| svm_rbf (C=1, RBF) | 0.807 | 0.676 | 0.884 | 0.33 ms | 2.5 MB |
| mlp (256 hidden) | 0.782 | 0.625 | 0.863 | 0.12 ms | 3.1 MB |

## Decision: svm_rbf
- Best on overall F1, on the weakest video and on AUC.
- Its extra latency (0.2 ms) and size (2.5 MB) are negligible next to the
  DINOv2-base backbone (hundreds of ms per image on CPU, ~350 MB in the image).

## What I accepted losing
- The simplicity, tiny size and interpretability of logistic regression.
- SVM prediction cost grows with the number of support vectors, so it gets
  slower as the training set grows.

## Caveat
- The test set is small (68 images, 4 videos). One image changes macro F1 by
  roughly 1.5 points, so the SVM vs MLP gap (0.025) is within noise. The gap
  to logistic regression (0.11) is not.

## Promotion rules (scripts/promote.py)
- Minimum F1 gain: 0.02. Smaller differences are about one test image, which is noise.
- Latency budget: p95 of the head <= 5 ms. Laptop p95 is 0.1-0.3 ms; CI runners
  are slower, so the budget leaves a wide safety margin while still blocking a
  head that is much slower.
