# CharLSTM Text Generation Evaluation Summary

## 1. Quantitative Benchmark Metrics
- **Validation Perplexity (PPL)**: 1.39
- **Total Samples Evaluated**: 150 (5 seeds x 3 temperatures x 10 runs)
- **Format Validity Rate**: 100.0% (matches specification update syntax)
- **Vocabulary Validity Rate**: 95.0% (words present in training corpus)
- **Catalog Relevance Rate**: 93.3% (contains target part + material/color)
- **Lexical Diversity**:
  - Distinct-1 (Unigrams): 0.0925
  - Distinct-2 (Bigrams): 0.2568
- **Training Duplicate Rate**: 0.0% (novel variations vs. exact memorization)

---

## 2. Temperature Comparison
| Temperature | Format Valid % | Vocab Valid % | Relevance % |
| :---: | :---: | :---: | :---: |
| **0.5 (Conservative)** | 100.0% | 99.7% | 98.0% |
| **0.8 (Optimal)** | 100.0% | 96.8% | 94.0% |
| **1.1 (Exploratory)** | 100.0% | 88.5% | 88.0% |

---

## 3. Qualitative Samples: Best vs. Worst

### Best 3 Samples (High Syntax Fidelity, Coherence & Material Plausibility)
| # | Temp | Generated Specification | Format | Vocab |
| :---: | :---: | :--- | :---: | :---: |
| 1 | 0.5 | `Update: chair legs changed to white cherry wood, glossy finish.` | True | 1.0 |
| 2 | 0.5 | `Update: chair legs changed to charcoal linen in charcoal walnut, satin finish.` | True | 1.0 |
| 3 | 0.5 | `Update: chair legs changed to charcoal walnut with brushed brass, satin finish.` | True | 1.0 |

### Worst 3 Samples (High Temperature Drift, Truncation, or Non-Standard Syntax)
| # | Temp | Generated Specification | Format | Vocab |
| :---: | :---: | :--- | :---: | :---: |
| 1 | 1.1 | `Update: chair legs changed to brushed barme wile, matte finish.` | True | 0.8 |
| 2 | 1.1 | `Spec update: cabinet handles to black lear #rest, finish.` | True | 0.7778 |
| 3 | 1.1 | `Update: sofa cushions updated to bouge cherry wiond chraler polished finish.` | True | 0.7273 |
