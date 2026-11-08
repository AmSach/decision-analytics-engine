# Real-World Global Superstore Case Study
### Empirical Benchmark on 9,426 Live Enterprise Retail Transactions

This folder demonstrates the **Decision Analytics Engine** deployed on the canonical **Global Superstore Enterprise Retail Dataset** (9,426 order rows across 2,703 accounts over a 4-year longitudinal operating horizon).

```
[ Raw Superstore CSV ] ──> [ UniversalExcelIngestor ] ──> [ Standardized Decision Stream ]
                                                                      │
              ┌──────────────────────────┬────────────────────────────┴────────────────────────────┐
              ▼                          ▼                                                         ▼
    [ DML Elasticity ]         [ CLV Equity Waterfall ]                                  [ Multi-Echelon MEIO ]
    Causal Beta = -0.03        746 Champions Driving                                     1,263 Categories
    (p-val < 0.001, SE 0.007)  $771,207.65 Forward Margin                                $646,081 Committed WC
```

---

## 1. Executive Deliverables Generated

All files in `./examples/outputs/` were generated directly from the live data via `python examples/run_superstore_benchmark.py`:

| Output Deliverable | File Path | Key Quantitative Findings |
|:---|:---|:---|
| **Causal Price Elasticity** | [`superstore_dml_elasticity.csv`](outputs/superstore_dml_elasticity.csv) | DML estimated unconfounded elasticity $\beta_{\text{DML}} = -0.03$ (SE: 0.007, 95% CI: `[-0.043, -0.016]`). Confirms extreme price insensitivity in core B2B office categories. |
| **Customer Equity Waterfall** | [`superstore_clv_waterfall.csv`](outputs/superstore_clv_waterfall.csv) | Top 746 **Champions** generate **$771,207.65** in 12-month forward expected net margin equity (77.8% of total portfolio value). |
| **DuPont 3-Tier ROE** | [`superstore_dupont_roe.csv`](outputs/superstore_dupont_roe.csv) | Total ROE of **113.75%**, decomposed into 26.15% net margin, 2.18x asset turnover, and 2.00x leverage. |
| **Cash Float Yield** | [`superstore_working_capital_float.csv`](outputs/superstore_working_capital_float.csv) | -44 days Cash Conversion Cycle (CCC) generates **$68,363.39/year** in risk-free float treasury yields, self-funding acquisition spend. |
| **Multi-Echelon MEIO Policy** | [`superstore_inventory_meio.csv`](outputs/superstore_inventory_meio.csv) | Solved safety stock, ROP, and EOQ for 1,263 categories, maintaining a 99.8%+ fill rate with **$646,081.87** working capital committed. |
| **Multivariate Anomaly Leakage**| [`superstore_anomaly_leakage_summary.csv`](outputs/superstore_anomaly_leakage_summary.csv) | Identified 268 operational anomalies (2.84% anomaly rate) via regularized Mahalanobis distance at $p < 0.01$. |
| **Network Fragility & Centrality**| [`superstore_network_vulnerability.csv`](outputs/superstore_network_vulnerability.csv) | Mapped systemic flow: Technology (PageRank: 0.1646, Fragility: 64.66) and Furniture (PageRank: 0.1456, Fragility: 51.72) drive top network throughput. |
| **Synthetic Control Lift** | [`superstore_synthetic_control_lift.csv`](outputs/superstore_synthetic_control_lift.csv) | Abadie Synthetic Control measured true quarterly promotional expansion lift of **+$7,321.58/quarter** against synthetic donor regions. |
| **Markowitz Portfolio Allocation**| [`superstore_markowitz_allocation.csv`](outputs/superstore_markowitz_allocation.csv) | Solved optimal Pareto capital allocation across Furniture, Office Supplies, and Technology under VaR constraints. |

---

## 2. Quick Reproduction

```bash
# Execute the real-world benchmark pipeline
python examples/run_superstore_benchmark.py
```
