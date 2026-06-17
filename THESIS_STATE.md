# THESIS_STATE.md
<!-- Update: YYYY-MM-DD | Paste into Claude Project before writing sessions -->

## Meta
- **Title (working):** Robustness of Surrogate Models for Biological Sequence Design under Distributional Shift
- **Uni:** JKU Linz | **Supervisor:** Johannes Schimunek
- **Datasets:** TFBind8 (DNA, 65,536 8-mers) · GB1 (protein fitness)
- **Surrogate trained on:** bottom 50% by fitness score

## Research Questions
1. How effectively do different optimizers find high-quality biological sequences?
2. How well does an MLP surrogate generalize under distributional shift (OOD)?

## Setup
| Component | Detail |
|-----------|--------|
| Surrogates | MLP, Random Forest (RF) |
| Optimizers | SMW (naive), RL (LSTM+MLP), GFlowNet (Jain et al. 2022) |
| OOD Splits | A (interpolation) · B (local extrapolation) · C (unseen clusters) |
| Metrics | Spearman ρ · MSE · Bias = mean(pred − true) |

## Current Results

### Surrogate Generalization
| Model | Dataset | Spearman ρ | MSE | Bias |
|-------|---------|-----------|-----|------|
| RF | TFBind8 | 0.37 | 0.046 | 0.134 |
| MLP | TFBind8 | 0.27 | 0.044 | 0.122 |
| RF | GB1 | 0.33 | 23.0 | 3.81 |
| MLP | GB1 | 0.65 | 14.6 | 2.94 |

**Key finding:** MLP > RF on GB1; RF slightly better on TFBind8 → dataset-dependent reversal (main novel result).
**Both models:** systematic positive bias (structural: trained on bottom 50%, no high-fitness calibration).

### Optimization
| Optimizer | Dataset | Max Oracle |
|-----------|---------|-----------|
| SMW | TFBind8 | 0.81 |
| RL (LSTM) | TFBind8 | 0.96 ± 0.03 |
| SMW | GB1 | 0.18 ± 0.04 |
| RL | GB1 | 🔄 pending |
| GFlowNet | GB1 | 🔄 pending |

## Written Sections (LaTeX — current drafts)

### ✅ 4.1.2 TFBind8 Surrogate Results — DONE
Key points covered: RF vs MLP comparison · positive bias explanation · MAE vs Hamming distance degradation · prediction variance (MLP σ=0.071 vs RF σ=0.033).

### 🔄 In Progress
- [ ] 4.1.3 GB1 Surrogate Results
- [ ] 4.2.x Optimization Results (waiting on RL GB1 + GFlowNet)

### ⬜ Not started
- [ ] Discussion / Interpretation
- [ ] Abstract
- [ ] Introduction (final version)

## Framing Notes (for writing)
- **Do NOT oversell** positive bias as surprising — it's a structural consequence
- **Lead with** dataset-dependent RF vs MLP reversal as the main interesting result
- **OOD severity** increases A → B → C; MAE monotonically increases with Hamming distance
- MLP extrapolates more aggressively (higher σ, higher max pred) — whether good or bad depends on optimization results (cross-ref Section 4.2)

## Next Steps
- [ ] Run RL on GB1
- [ ] Run GFlowNet experiments
- [ ] Write 4.1.3 GB1 section
- [ ] Write 4.2 Optimization Results

## Repo
- Private GitHub repo (link omitted for security)
- Main scripts: `train_surrogate.py` · `optimize.py` · `evaluate_ood.py` (update names as needed)