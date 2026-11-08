"""
Universal Adaptive Excel and CSV Ingestion Engine.
Automatically ingests heterogeneous, messy Excel workbooks (.xlsx, .xls) and CSVs:
1. Detects header rows past metadata banners and comment headers (#, //).
2. Performs fuzzy semantic column resolution (matching Cust_ID, Net Sales, COGS, etc.).
3. Strips currency symbols, formatting noise, and converts dates across formats.
4. Auto-derives missing metrics (Revenues = Price * Qty, Cohorts = min(Order Date)).
5. Emits harmonized DataFrames directly into the decision analytics suite.
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd


class UniversalExcelIngestor:
    """
    Intelligent Schema Normalizer for Arbitrary Business Data.
    """

    # Semantic synonym dictionaries with strict boundary matching for sub-tokens
    SYNONYM_MAP = {
        "customer_id": [
            r"cust.*id", r"customer.*id", r"client.*id", r"user.*id", 
            r"account.*id", r"\bbuyer\b", r"\bcustomer\b", r"\bclient\b", r"\baccount\b"
        ],
        "order_date": [
            r"order.*date", r"trans.*date", r"txn.*date", r"invoice.*date", 
            r"purchase.*date", r"\bdate\b", r"time.*stamp", r"\bdatetime\b"
        ],
        "sku": [
            r"\bsku\b", r"product.*id", r"item.*id", r"product.*code", 
            r"item.*code", r"product.*name", r"item.*name", r"description", r"\bitem\b"
        ],
        "effective_price": [
            r"selling.*price", r"unit.*price", r"effective.*price", 
            r"net.*price", r"selling.*rate", r"\bprice\b", r"\brate\b"
        ],
        "quantity": [
            r"quant.*", r"\bqty\b", r"\bunits\b", r"volume.*sold", r"\bvolume\b", r"\bcount\b"
        ],
        "gross_revenue": [
            r"gross.*rev.*", r"net.*sales", r"total.*sales", r"revenue", 
            r"\bsales\b", r"total.*amount", r"order.*amount", r"invoice.*amt", r"extended.*price"
        ],
        "cogs": [
            r"\bcogs\b", r"unit.*cost", r"prod.*cost", r"cost.*goods", 
            r"\bcost\b", r"\bexpense\b", r"item.*cost"
        ],
        "discount_pct": [
            r"disc.*pct", r"discount.*rate", r"disc.*rate", 
            r"promo.*disc.*", r"discount", r"\bdisc\b", r"promo.*pct", r"\brebate\b"
        ],
        "channel": [
            r"\bchannel\b", r"\bsource\b", r"\bmedium\b", r"utm_source", 
            r"acquisition.*channel", r"campaign.*channel", r"\bplatform\b"
        ],
    }

    def _clean_numeric(self, series: pd.Series) -> pd.Series:
        """Cleans currency strings, percentages, and commas into pure float."""
        if pd.api.types.is_numeric_dtype(series):
            return series.astype(float).fillna(0.0)
        
        cleaned = (
            series.astype(str)
            .str.replace(r"[\$,₹,€,£,\s]", "", regex=True)
            .str.replace(r"%", "", regex=True)
        )
        return pd.to_numeric(cleaned, errors="coerce").fillna(0.0)

    def _clean_datetime(self, series: pd.Series) -> pd.Series:
        """Parses mixed date formats including Excel serial timestamps."""
        # Check if Excel serial integer format (e.g. 44927 for year 2023)
        if pd.api.types.is_numeric_dtype(series) and series.mean() > 30000:
            return pd.to_datetime(series, unit="D", origin="1899-12-30", errors="coerce")
        return pd.to_datetime(series, format="mixed", errors="coerce")

    def _match_column(self, col_name: str) -> str | None:
        """Fuzzy matches raw column name against semantic synonym patterns."""
        normalized = col_name.strip().lower().replace(" ", "_")
        for standard_key, patterns in self.SYNONYM_MAP.items():
            for p in patterns:
                if re.search(p, normalized):
                    return standard_key
        return None

    def ingest_file(self, file_path: str | Path, sheet_name: str | int | None = 0) -> pd.DataFrame:
        """
        Reads any Excel or CSV file and normalizes into standard schema.
        """
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # 1. Load Raw DataFrame with comment handling
        if p.suffix.lower() in [".xlsx", ".xls"]:
            raw_df = pd.read_excel(p, sheet_name=sheet_name)
        elif p.suffix.lower() == ".csv":
            raw_df = pd.read_csv(p, comment="#", skip_blank_lines=True)
        else:
            raise ValueError(f"Unsupported file format '{p.suffix}'. Must be .xlsx, .xls, or .csv")

        # 2. Detect real header if top rows are blank or company metadata
        if raw_df.columns.str.contains(r"^Unnamed").mean() > 0.6:
            for idx, row in raw_df.iterrows():
                valid_count = row.dropna().count()
                if valid_count >= 3:
                    raw_df.columns = row.values
                    raw_df = raw_df.iloc[idx + 1:].reset_index(drop=True)
                    break

        # 3. Map Columns
        column_mapping = {}
        for col in raw_df.columns:
            std_name = self._match_column(str(col))
            if std_name and std_name not in column_mapping.values():
                column_mapping[col] = std_name

        harmonized = raw_df.rename(columns=column_mapping).copy()

        # 4. Clean and Standardize Data Types
        if "customer_id" in harmonized.columns:
            harmonized["customer_id"] = harmonized["customer_id"].astype(str).str.strip()

        if "order_date" in harmonized.columns:
            harmonized["parsed_order_date"] = self._clean_datetime(harmonized["order_date"])

        for num_col in ["effective_price", "quantity", "gross_revenue", "cogs", "discount_pct"]:
            if num_col in harmonized.columns:
                harmonized[num_col] = self._clean_numeric(harmonized[num_col])

        # 5. Impute Missing Relational Fields
        # Revenue = Price * Quantity if missing
        if "gross_revenue" not in harmonized.columns and "effective_price" in harmonized.columns and "quantity" in harmonized.columns:
            harmonized["gross_revenue"] = (harmonized["effective_price"] * harmonized["quantity"]).round(2)
        elif "effective_price" not in harmonized.columns and "gross_revenue" in harmonized.columns and "quantity" in harmonized.columns:
            harmonized["effective_price"] = np.where(
                harmonized["quantity"] > 0, 
                (harmonized["gross_revenue"] / harmonized["quantity"]).round(2), 
                0.0
            )

        if "quantity" not in harmonized.columns:
            harmonized["quantity"] = 1

        if "discount_pct" not in harmonized.columns:
            harmonized["discount_pct"] = 0.0
        else:
            # If discount is in 0-100 range, normalize to 0.0-1.0
            if harmonized["discount_pct"].max() > 1.0:
                harmonized["discount_pct"] = (harmonized["discount_pct"] / 100.0).round(4)

        if "cogs" not in harmonized.columns and "gross_revenue" in harmonized.columns:
            # Conservative default: 35% COGS benchmark
            harmonized["cogs"] = (harmonized["gross_revenue"] * 0.35).round(2)

        if "gross_margin" not in harmonized.columns and "gross_revenue" in harmonized.columns and "cogs" in harmonized.columns:
            harmonized["gross_margin"] = (harmonized["gross_revenue"] - harmonized["cogs"]).round(2)

        if "sku" not in harmonized.columns:
            harmonized["sku"] = "DEFAULT_SKU"

        if "channel" not in harmonized.columns:
            harmonized["channel"] = "Direct"

        if "customer_id" not in harmonized.columns:
            harmonized["customer_id"] = [f"CUST-{i + 1:05d}" for i in range(len(harmonized))]

        if "transaction_id" not in harmonized.columns:
            harmonized["transaction_id"] = [f"TX-{i + 1:07d}" for i in range(len(harmonized))]

        # Automatically derive customer cohort month from parsed datetime
        if "parsed_order_date" in harmonized.columns and harmonized["parsed_order_date"].notna().sum() > 0:
            first_orders = harmonized.groupby("customer_id")["parsed_order_date"].min().dt.strftime("%Y-%m")
            harmonized["cohort_month"] = harmonized["customer_id"].map(first_orders)
            harmonized["order_date"] = harmonized["parsed_order_date"].dt.strftime("%Y-%m-%d")
            harmonized.drop(columns=["parsed_order_date"], inplace=True)

        return harmonized
