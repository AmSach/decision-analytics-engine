# Decision Analytics Engine
### Advanced Econometric Modeling, Adversarial Game Theory & Palantir-Grade Operational Ontology

[![CI Test Suite](https://github.com/AmSach/decision-analytics-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/AmSach/decision-analytics-engine)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Unit Tests: 26 Passed](https://img.shields.io/badge/tests-26%20passed-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)

A boardroom-grade enterprise decision architecture engineered for C-suite executives, corporate strategy leaders, and graduate business school quants. Moving past descriptive aggregates and vanity dashboards, this engine deploys causal econometric identification, adversarial game-theoretic pricing, polycentric commons governance, synthetic quasi-experiments, Markowitz capital frontiers, and an enterprise operational ontology.

```
                                 [ Multi-Channel Order & Path Streams ]
                                                   │
         ┌─────────────────────────┬───────────────┴───────────────┬─────────────────────────┐
         ▼                         ▼                               ▼                         ▼
 [ Causal DML Elasticity ]   [ Level-k Adversarial ]      [ Ostrom Commons CPR ]   [ Spectral Hypergraph ]
  Neyman-Orthogonal Score     Quantal Response (QRE)       Tragedy-of-the-Commons   Perron-Frobenius Katz
  Robinson Residualization    Price War Defection Defense  CAC Kill-Gate Throttle   Algebraic Connectivity
         │                         │                               │                         │
         ├─────────────────────────┼───────────────────────────────┼─────────────────────────┤
         ▼                         ▼                               ▼                         ▼
 [ Synthetic Control Lift ]  [ Markowitz Frontier ]       [ Universal Excel ]      [ Stochastic CCC Float ]
  Abadie Donor Pool Match     Quadratic Programming        Fuzzy Column Resolver    Negative Working Capital
  Counterfactual Lift Est.    VaR 95% Ad Allocation        Messy Workbook Ingest    Treasury Yield Generation
         │                         │                               │                         │
         └─────────────────────────┴───────────────┬───────────────┴─────────────────────────┘
                                                   ▼
                                [ Palantir-Grade Decision Ontology ]
                                 - Live Operational Risk Telemetry
                                 - Counterfactual War-Gaming Sandbox
                                 - Blast Radius & Remediation Gates
```

---

## 1. The Intuitive Mental Models (How the Data Moves)

To make sound executive decisions, leaders must be able to picture the physical mechanics behind the equations:

### A. The Smudged Mirror vs Clean Lens (Double Machine Learning)
Imagine selling umbrellas. On rainy days, sales skyrocket even at $30. On sunny days, sales hit zero even at $5. A naive algorithm assumes higher prices cause higher sales because it cannot see the rain.  
**How DML Works:** Two robotic scrubbers clean the data. Scrubber 1 scrubs away customer wealth and seasonal demand. Scrubber 2 scrubs away promotional pricing noise. What remains is a crystal-clear lens revealing true, unconfounded price sensitivity.

### B. The Chessboard of Price Wars (Level-k & Quantal Response Equilibrium)
Standard pricing treats competitors as frozen statues. In reality, if you cut prices to steal share, rivals retaliate. Both firms bleed out in a destructive race to the bottom.  
**How Game Theory Solves This:** Level-1 shoots where the rival stands today. Level-2 anticipates your move and steps aside. Level-3 (Our Engine) fires where the rival is about to jump. The engine calculates the **Nash/QRE equilibrium ($81.47)**, maximizing cash margin while making competitor price attacks mathematically suicidal.

### C. The Overfished Pond & Secret Side Canal (Elinor Ostrom Commons Governance)
Shared ad auction pools (Google/Meta Ads) are an open-access fishing lake. When rival brands get greedy and flood the water with bids, the lake gets muddy and Cost-Per-Acquisition (CAC) explodes from $20 to $90 (Hardin's Tragedy of the Commons).  
**What Our Engine Does:** A watchtower tracks the Commons Depletion Index. The instant congestion triggers an automated kill-gate, your budget quietly pulls out of the war and slips down a private canal (high-ROI email and organic channels), acquiring customers for $15 while competitors incinerate capital.

### D. The Glowing Pinball Machine (Spectral Graph Attribution)
Customer journeys are pinballs bouncing across touchpoints: Instagram $\to$ Tech Blog $\to$ Retargeting $\to$ Email $\to$ Purchase. Last-click attribution gives 100% of the glory to Email just because it was touched last.  
**What Spectral Graph Theory Does:** It treats the marketing ecosystem as an electrical grid. Using Perron-Frobenius leading eigenvectors and Katz attenuation, it measures structural network bridges, proving which early channels created the momentum so you never cut the fuel line.

### E. The Parallel Universe Twin (Abadie Synthetic Control)
When launching a regional marketing campaign or price test, you cannot run an A/B test in the past. You don't have a time machine.  
**What Synthetic Control Does:** It blends untreated donor regions (e.g. 40% South + 35% East + 25% West) into a virtual "Doppelganger Clone" that mirrored your test region before the campaign. By comparing your real region against its synthetic twin during the campaign, the exact incremental lift is isolated without speculation.

### F. The Wall Street Balancing Scale (Markowitz Capital Frontier)
Pouring all ad spend into the single highest-return channel causes channel exhaustion and high variance.  
**How Markowitz Solves This:** It treats marketing channels and SKU inventory like financial assets. It solves the quadratic Pareto frontier under 95% Value-at-Risk (VaR), allocating capital to achieve optimal portfolio returns while strictly hedging downside concentration risk.

### G. Free Rocket Fuel Cash Float (Negative Working Capital CCC)
Customers pay immediately on Day 0. Inventory turns in 14 days. Suppliers are paid on Day 60.  
**The 44-Day Gap:** You hold customer cash in your bank for 44 days before owing suppliers. For a $5M enterprise, you permanently float $600,000 of other people's money. The engine simulates risk-free money-market Treasury yields ($11,523/year) that self-fund ad spend before bills come due.

---

## 2. Palantir-Grade Enterprise Architecture

### A. Enterprise Decision Ontology & Operational Digital Twin (`src/enterprise_ontology.py`)
Maps heterogeneous business tables into an interconnected relational object graph linking:
$$\text{Customers} \longleftrightarrow \text{Transactions} \longleftrightarrow \text{SKUs} \longleftrightarrow \text{Marketing Channels} \longleftrightarrow \text{Working Capital}$$
Tracks real-time operational risk telemetry and issues automated trigger actions:
* Churn Hazard Alerts: Monitors Kaplan-Meier survival drops at Day 30-45.
* Elasticity Fragility: Detects SKU concentration risks exceeding 40% of revenue.
* Liquidity Telemetry: Monitors whether operational float covers baseline debt-service hurdles.

### B. Counterfactual War-Gaming Scenario Sandbox (`src/scenario_sandbox.py`)
Enables executives to fork business states and simulate multi-vector macroeconomic shocks:
* Rival Price Attacks (-15% competitor discounting)
* Supply Chain Inflation (+20% COGS jump)
* Working Capital Term Reductions (-15 days DPO compression)
Calculates the exact P&L margin blast radius and issues board-level remediation protocols.

---

## 3. Real-World Benchmark on 9,426 Live Transactions (Global Superstore Case Study)

The engine was benchmarked on the canonical **Global Superstore Enterprise Retail Dataset** (9,426 orders across 2,703 accounts over 4 years). All outputs are published in [`examples/outputs/`](examples/outputs/):

| Real-World Deliverable | Output Path | Key Empirical Results |
|:---|:---|:---|
| **Causal Price Elasticity** | [`superstore_dml_elasticity.csv`](examples/outputs/superstore_dml_elasticity.csv) | DML estimated unconfounded price elasticity $\beta_{\text{DML}} = -0.03$ (SE: 0.007, 95% CI: `[-0.043, -0.016]`). Confirmed high price-inelasticity across enterprise office lines. |
| **Customer Equity Waterfall** | [`superstore_clv_waterfall.csv`](examples/outputs/superstore_clv_waterfall.csv) | 746 **Champions** deliver **$771,207.65** in 12-month forward expected net margin equity (77.8% of total portfolio equity). |
| **DuPont 3-Tier ROE** | [`superstore_dupont_roe.csv`](examples/outputs/superstore_dupont_roe.csv) | Total ROE of **113.75%**, decomposed into 26.15% net profit margin, 2.18x asset turnover, and 2.00x equity leverage. |
| **Stochastic Cash Float Yield** | [`superstore_working_capital_float.csv`](examples/outputs/superstore_working_capital_float.csv) | -44 days Cash Conversion Cycle (CCC) generated **$68,363.39/year** in risk-free treasury float returns. |
| **Palantir Decision Ontology** | [`superstore_ontology_telemetry.json`](examples/outputs/superstore_ontology_telemetry.json) | Board Alert: *SUSTAINABLE RUNWAY: Deploy capital into Markowitz-optimized acquisition channels.* |
| **War-Gaming Blast Radius** | [`superstore_war_gaming_blast_radius.csv`](examples/outputs/superstore_war_gaming_blast_radius.csv) | Macro Shock (15% rival price drop + 20% freight cost surge) caused **$-1,288,284.70 (-22.15%)** margin erosion, triggering the *Critical Defense Kill-Gate*. |
| **Synthetic Control Lift** | [`superstore_synthetic_control_lift.csv`](examples/outputs/superstore_synthetic_control_lift.csv) | Abadie Synthetic Control isolated true quarterly promotional expansion lift of **+$7,321.58/quarter** against donor regions. |

---

## 4. Universal Adaptive Excel Ingestion (`src/excel_universal_ingestor.py`)

Handles chaotic, unstandardized business Excel workbooks (`.xlsx`, `.xls`, `.csv`) with zero manual cleanup:

```python
from src.excel_universal_ingestor import UniversalExcelIngestor

# Ingest any messy client workbook in 1 line
ingestor = UniversalExcelIngestor()
clean_df = ingestor.ingest_file("path/to/messy_client_data.xlsx")

# Automatically resolves column synonyms (Cust_ID, Selling Rate, Net Sales)
# Strips currency signs ('$', '₹', '%', commas) and derives customer cohort months
print(clean_df.head())
```

---

## 5. Repository Architecture

```
decision-analytics-engine/
├── data/                             <- Generated transaction datasets and pipeline CSV outputs
├── examples/
│   ├── raw_online_data/              <- Raw Global Superstore benchmark dataset (9,426 rows)
│   ├── outputs/                      <- Real-world benchmark CSV and JSON deliverables
│   ├── run_superstore_benchmark.py   <- Benchmark execution script on live retail data
│   └── README.md                     <- Detailed case study writeup and empirical findings
├── src/
│   ├── __init__.py
│   ├── data_generator.py             <- Multi-channel transaction and path synthesis
│   ├── excel_universal_ingestor.py   <- Universal messy Excel/CSV schema harmonizer
│   ├── enterprise_ontology.py        <- Palantir-style operational ontology digital twin
│   ├── scenario_sandbox.py           <- Counterfactual war-gaming shock simulator
│   ├── cohort_retention.py           <- Longitudinal cohort and Kaplan-Meier churn hazard
│   ├── price_elasticity.py           <- Econometric log-linear demand regression models
│   ├── causal_elasticity_dml.py      <- Double / Debiased Machine Learning (DML) engine
│   ├── adversarial_game_pricing.py   <- Level-k and Logit QRE competitive pricing solver
│   ├── ostrom_commons_governance.py  <- Elinor Ostrom CPR bidding war governance engine
│   ├── synthetic_control.py          <- Abadie Synthetic Control quasi-experiment engine
│   ├── portfolio_markowitz.py        <- Markowitz quadratic programming capital allocator
│   ├── spectral_attribution.py       <- Perron-Frobenius and Katz spectral graph attribution
│   ├── stochastic_float_engine.py    <- DuPont 3-tier ROE & negative CCC float simulation
│   ├── clv_model.py                  <- RFM portfolio segmentation and forward CLV equity
│   ├── marketing_attribution.py      <- First-order Markov chain removal-effect attribution
│   └── dashboard.py                  <- Interactive Streamlit executive decision cockpit
├── tests/
│   ├── __init__.py
│   ├── test_analytics.py             <- Baseline mathematical invariant tests (11 passing)
│   ├── test_advanced_algorithms.py   <- Game-theory and DML invariant tests (8 passing)
│   ├── test_excel_and_new_algorithms.py <- Synthetic Control, Markowitz & Excel tests (3 passing)
│   └── test_palantir_suite.py        <- Ontology and scenario war-gaming tests (4 passing)
├── run_pipeline.py                   <- Master execution orchestrator producing all deliverables
├── requirements.txt                  <- Python dependency specifications
└── README.md                         <- Executive documentation and architectural guide
```

---

## 6. Verification & Test Suite (26/26 Passing)

Every module is governed by strict invariant unit tests:

```bash
python -m pytest -v tests/
```

### Verified Invariants:
* `TestCausalElasticityDML`: Neyman-orthogonal score consistency, negative causal elasticity, valid asymptotic standard error bounds.
* `TestAdversarialGamePricing`: Best-response profitability bounds, QRE probability simplex conservation, fixed-point convergence.
* `TestOstromCommonsGovernance`: Pool congestion monotonicity, Ostrom CAC dominance over unregulated commons defection spirals.
* `TestSpectralAttribution`: Perron-Frobenius leading eigenvector normalization, non-negative Laplacian Fiedler algebraic connectivity.
* `TestSyntheticControlEngine`: Donor weights non-negativity and convex combination ($\sum W = 1.0$), post-intervention causal lift detection.
* `TestMarkowitzPortfolioOptimizer`: Budget conservation, channel allocation cap bounds, positive Sharpe ratio, non-negative 95% VaR.
* `TestUniversalExcelIngestor`: Multi-format currency parsing, fuzzy synonym column mapping, automatic cohort derivation.
* `TestEnterpriseDecisionOntology`: Node binding consistency, SKU entity queries, automated action triggers.
* `TestScenarioSandboxBlastRadius`: Multi-vector shock propagation, negative CCC float delta, critical defense kill-gate activation.
* `TestStochasticFloatEngine`: Exact 3-tier DuPont identity reconciliation, non-decreasing cumulative float yield.
* `TestCohortRetentionInvariants`: Period 0 retention normalization, Kaplan-Meier monotone survival, valid conditional hazard rates.

---

## 7. Quickstart & Execution

```bash
# 1. Clone repository
git clone https://github.com/AmSach/decision-analytics-engine.git
cd decision-analytics-engine

# 2. Run test suite (26 passing tests)
python -m pytest -v tests/

# 3. Execute master pipeline (Generates all synthetic deliverables)
python run_pipeline.py

# 4. Execute live Global Superstore real-world benchmark
python examples/run_superstore_benchmark.py

# 5. Launch interactive executive cockpit
streamlit run src/dashboard.py
```

---

## 8. License
MIT License. Developed for graduate business school decision modeling, enterprise strategic planning, and algorithmic data analytics.
