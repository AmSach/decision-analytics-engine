"""
Centralized Repository of Model Tooltips and Decision Rules.
Provides rich, accurate, commercial-grade hover information cards for every
built-in algorithm, econometric model, and decision metric.
"""

from typing import Dict, Any

MODEL_TOOLTIPS: Dict[str, Dict[str, str]] = {
    "causal_elasticity": {
        "title": "Causal Price Elasticity & Profit Optimization",
        "plain_english": (
            "Measures how much customer purchase volume drops when you change price, "
            "controlling for promotional discounts and seasonal spikes so you do not "
            "confuse discounts with true customer price sensitivity."
        ),
        "formula": "Q_d = A * P^(beta), where beta = d(ln Q) / d(ln P). Optimal Price P* = Cost / (1 + 1/beta)",
        "decision_rule": (
            "If |beta| < 1.0 (Inelastic): You are underpricing. Hike price immediately to harvest gross margin without losing significant volume. "
            "If |beta| > 1.0 (Elastic): Price cuts increase gross revenue, but watch unit margins. Use P* to set the profit-maximizing ticket price."
        ),
        "example": "Beta = -0.74 means a 10% price increase reduces volume by only 7.4%, generating net positive margin expansion."
    },
    "cash_flow_at_risk": {
        "title": "Cash Flow at Risk (CFaR) 5,000-Path Monte Carlo",
        "plain_english": (
            "Simulates 5,000 future operating cash flow scenarios under correlated shocks "
            "across revenue volume, COGS inflation, operating expenses, and customer payment float delays."
        ),
        "formula": "Simulated Cash Flow CF_t = Rev_t - COGS_t - Opex_t - Delta(NWC_t). Correlated via Cholesky decomposition L L^T = Sigma.",
        "decision_rule": (
            "Examine the 95% Value at Risk (VaR) floor. If the 95% worst-case cash flow breaches your bank debt covenant floor, "
            "the firm must secure a revolving credit facility or build a liquid buffer before initiating growth capital expenditures."
        ),
        "example": "95% VaR floor of $420,000 against a $500,000 loan covenant implies a 3.4% probability of default, requiring an $80,000 liquidity buffer."
    },
    "anomaly_sweep": {
        "title": "Multivariate Anomaly & Margin Leak Detector",
        "plain_english": (
            "Audits transactions for statistical outliers using robust Z-scores (median absolute deviation) "
            "and Interquartile Range (IQR), flagging pricing errors, extreme discounting, and negative gross margins."
        ),
        "formula": "Modified Z = 0.6745 * (x - Median) / MAD. Outlier if |Z| > 3.0 or x < Q1 - 1.5*IQR or x > Q3 + 1.5*IQR.",
        "decision_rule": (
            "Instantly quarantine flagged transactions with negative gross margin or abnormal volume. "
            "Verify whether these are rogue discounts, ERP data entry typos, or fraudulent duplicate orders before ledger close."
        ),
        "example": "Flags 12 transactions where gross margin is negative due to stacked coupons applied simultaneously."
    },
    "inventory_meio": {
        "title": "Multi-Echelon Inventory Optimization (MEIO)",
        "plain_english": (
            "Calculates the exact safety stock and reorder point needed for each SKU to prevent stockouts "
            "while minimizing working capital locked up in warehouse storage."
        ),
        "formula": "Safety Stock SS = z_alpha * sqrt(L * sigma_D^2 + D^2 * sigma_L^2). Reorder Point ROP = D * L + SS.",
        "decision_rule": (
            "For fast-moving SKUs (Category A), maintain a 98% service level. For slow-moving tail SKUs (Category C), "
            "drop to 90% service level to free up working capital. Trigger purchase orders when inventory reaches ROP."
        ),
        "example": "Sets ROP at 240 units with 45 units of safety stock, reducing holding costs by 18% compared to naive 30-day supply rules."
    },
    "cohort_churn": {
        "title": "Cohort Retention Matrix & Churn Hazard Function",
        "plain_english": (
            "Tracks customer cohorts grouped by acquisition month to see what percentage return in subsequent months, "
            "revealing whether retention curves flatten into a sustainable recurring customer base."
        ),
        "formula": "Retention Rate R_t = Active_t / Cohort_0. Churn Hazard h_t = (R_{t-1} - R_t) / R_{t-1}.",
        "decision_rule": (
            "If retention continues decaying toward 0% without leveling off, the product has a leaky bucket problem: pause top-of-funnel ad spend "
            "and fix onboarding. If retention flattens (e.g. at 35% after month 3), you have achieved product-market fit."
        ),
        "example": "Month 1 retention is 42%, Month 2 is 38%, Month 3 is 37%. The flattening curve validates scalable customer acquisition."
    },
    "markov_attribution": {
        "title": "Markov Multi-Touch Marketing Attribution",
        "plain_english": (
            "Models customer buyer journeys as an ergodic state machine. Measures the true removal effect of each channel "
            "(Paid Search, Email, Social) to quantify how conversions drop when a channel is removed from the mix."
        ),
        "formula": "Transition Matrix P, where P_ij = Transitions(i -> j) / Total(i). Removal Effect r_k = (P_base - P_{-k}) / P_base.",
        "decision_rule": (
            "Compare Markov attribution weights against naive Last-Touch. If Paid Search is credited 50% in Last-Touch but only 20% in Markov, "
            "reallocate ad spend toward upper-funnel discovery channels like Organic Content or Email nurture."
        ),
        "example": "Email shows a 34% removal effect despite low first-touch attribution, proving it is the essential conversion catalyst."
    },
    "mixed_logit_cannibalization": {
        "title": "Mixed Logit Diversion & Product Cannibalization",
        "plain_english": (
            "Models heterogeneous consumer choices across product tiers. Calculates the Diversion Ratio: when you hike price on Tier A, "
            "how much demand shifts to your own Tier B versus defecting to external competitors."
        ),
        "formula": "Diversion Ratio D_{j->k} = - [d S_k / d P_j] / [d S_j / d P_j]. Internal Share = sum_{k in Firm} D_{j->k}.",
        "decision_rule": (
            "If internal cannibalization exceeds 70%, price increases on entry tiers are safe because defecting customers upgrade to higher-margin tiers. "
            "If external leakage exceeds 40%, price increases will destroy portfolio market share."
        ),
        "example": "A 10% price hike on Standard tier results in 78% of defecting buyers upgrading to Pro tier, netting an overall profit increase."
    },
    "bandit_experiment": {
        "title": "Bayesian Multi-Armed Bandit Experimentation",
        "plain_english": (
            "Dynamically routes traffic between promotional offers or website variants using Beta-Binomial Thompson Sampling. "
            "Unlike static 50/50 A/B testing, it earns while learning by sending more visitors to the winning variant in real time."
        ),
        "formula": "Beta distribution prior Beta(alpha, beta). Posterior update: alpha <- alpha + conversions, beta <- beta + (traffic - conversions).",
        "decision_rule": (
            "Terminate static 50/50 A/B tests immediately. Adopt Thompson Sampling to minimize cumulative regret. "
            "When posterior probability of superiority exceeds 95%, declare the winner and lock in the 100% allocation."
        ),
        "example": "Captures 90% fewer lost conversions during the test period compared to a traditional 60-day static split test."
    },
    "network_fragility": {
        "title": "Supply Chain & Category Network Fragility",
        "plain_english": (
            "Maps the co-occurrence, transactional flow, or supplier dependency network into a directed graph. "
            "Computes PageRank and hub centrality to identify single points of failure."
        ),
        "formula": "Google Matrix G = d * M + (1 - d)/N * E. Node Centrality C_i = sum_j G_ij * C_j.",
        "decision_rule": (
            "Identify nodes with top 5% centrality scores. If a single supplier or SKU category accounts for >40% of graph centrality, "
            "enforce dual-sourcing contracts immediately to eliminate operational bottleneck fragility."
        ),
        "example": "Supplier #104 has a 0.38 PageRank centrality; an outage there halts 64% of downstream order fulfillment."
    },
    "knapsack_allocator": {
        "title": "Knapsack Resource & Budget Optimization",
        "plain_english": (
            "Solves the 0/1 integer linear programming problem to maximize total return or strategic impact "
            "subject to a hard budget ceiling and mutually exclusive project constraints."
        ),
        "formula": "Maximize sum(r_i * x_i) subject to sum(c_i * x_i) <= Budget, where x_i in {0, 1}.",
        "decision_rule": (
            "Replaces subjective managerial pet projects with mathematical capital allocation. "
            "Select the exact subset of initiatives that generates the maximum aggregate ROI without overrunning the Capex budget."
        ),
        "example": "Selects Projects 1, 4, 7, and 9 for $1.2M total cost, yielding $3.4M in expected value vs $2.1M from greedy prioritization."
    }
}


NODE_TOOLTIPS: Dict[str, Dict[str, str]] = {
    "source_csv": {
        "title": "CSV / TSV File Loader",
        "plain_english": "Ingests delimited text files into an in-memory high-speed Arrow table with automatic type detection.",
        "formula": "Parses RFC 4180 CSV specifications with UTF-8 encoding.",
        "decision_rule": "Use as the starting node for raw transactional logs or external dumps.",
        "example": "Loads transactions_2026.csv (24,000 rows in 12ms)."
    },
    "source_excel": {
        "title": "Excel Workbook Loader",
        "plain_english": "Loads Microsoft Excel workbooks (.xlsx, .xls) and allows selecting individual sheets or unioning multiple tables.",
        "formula": "Extracts raw cell matrices, preserving numeric precision and date formats.",
        "decision_rule": "Ideal for ingesting multi-tab operational models or financial budgets.",
        "example": "Pulls 'Q3_Actuals' sheet from 'Corporate_Forecast.xlsx'."
    },
    "source_synthetic": {
        "title": "Enterprise Sample Data Generator",
        "plain_english": "Generates a realistic e-commerce enterprise dataset with orders, customers, categories, pricing, and margins.",
        "formula": "Deterministic pseudo-random generation with log-normal price distributions and Poisson order counts.",
        "decision_rule": "Use for immediate 1-click testing, demonstrations, or sandbox simulation.",
        "example": "Spawns 5,000 transaction records with verified financial consistency."
    },
    "transform_filter": {
        "title": "Row Filter & Query Node",
        "plain_english": "Filters rows based on boolean conditions (e.g. price > 50, status == 'Active').",
        "formula": "Subset dataframe where Predicate(row) == True.",
        "decision_rule": "Eliminate cancelled orders, test accounts, or specific categories before running downstream analytics.",
        "example": "Filters for orders with 'gross_margin > 0' and 'tier == Enterprise'."
    },
    "transform_columns": {
        "title": "Column Selection & Projection",
        "plain_english": "Selects, reorders, or drops specific columns to streamline downstream computation.",
        "formula": "Matrix column projection: X[:, [j_1, j_2, ... j_k]].",
        "decision_rule": "Keep tables lean by removing redundant identifier columns before feeding into clustering or modeling.",
        "example": "Keeps 'order_date', 'customer_id', 'gross_revenue', 'gross_margin'."
    },
    "transform_groupby": {
        "title": "Group By & Aggregate Node",
        "plain_english": "Groups data by categorical keys (e.g. Region, Category, Month) and aggregates metrics (Sum, Mean, Count, Max).",
        "formula": "Split-Apply-Combine operator: Group(K) -> Aggregate(f(X)).",
        "decision_rule": "Roll up granular transaction logs into customer-level or monthly executive summaries.",
        "example": "Groups by 'category' and calculates sum of 'gross_revenue' and mean of 'gross_margin'."
    },
    "transform_dedup": {
        "title": "Deduplication & Entity Matching",
        "plain_english": "Identifies and removes exact and near-duplicate records using key columns or composite hashes.",
        "formula": "Hash(c_1, c_2, ... c_k) uniqueness check.",
        "decision_rule": "Run on raw CRM exports to prevent double-counting customer counts or revenue.",
        "example": "Eliminates 38 duplicate order rows generated by double-submission retries."
    },
    "transform_formula": {
        "title": "Computed Formula Column",
        "plain_english": "Appends a new column computed using mathematical expressions or column arithmetic.",
        "formula": "Supports Excel-like formulas (=A*B) and Python expressions (revenue - cogs).",
        "decision_rule": "Create custom derived metrics like Net Margin Ratio, Revenue per User, or Days Since Purchase.",
        "example": "Adds 'margin_pct' = (gross_margin / gross_revenue) * 100."
    },
    "export_csv": {
        "title": "Export Cleaned CSV",
        "plain_english": "Saves the current node dataset to a clean, standard CSV file on disk.",
        "formula": "Serializes in-memory table to UTF-8 delimited text.",
        "decision_rule": "Use to hand off cleaned datasets to external production systems.",
        "example": "Exports cleaned_orders.csv ready for database loading."
    },
    "export_excel": {
        "title": "Export Formatted Excel Workbook",
        "plain_english": "Saves datasets and analysis summary sheets into an Excel workbook (.xlsx).",
        "formula": "Structured workbook generation with header formatting and autosized columns.",
        "decision_rule": "Generate executive-ready workbooks for CFO or boardroom review.",
        "example": "Exports Decision_Summary.xlsx with raw data and aggregated analytics."
    }
}
