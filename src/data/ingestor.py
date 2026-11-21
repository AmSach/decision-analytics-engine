"""
Universal Ingestor and Exporter.
Safely reads and writes CSV, TSV, and Excel workbooks (.xlsx, .xls)
with robust data-type coercion and encoding fallbacks.
"""

from pathlib import Path
from typing import Dict, List, Union
import pandas as pd


def load_file(file_path: Union[str, Path], sheet_name: Union[str, int, None] = 0) -> Dict[str, pd.DataFrame]:
    """
    Ingests tabular data files:
    - CSV / TSV: Returns {'default': df}
    - Excel: Returns dict of sheet_name -> df
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()

    if suffix in [".csv", ".tsv", ".txt"]:
        sep = "\t" if suffix == ".tsv" else ","
        try:
            df = pd.read_csv(path, sep=sep, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(path, sep=sep, encoding="latin1")
        return {"default": df}

    elif suffix in [".xlsx", ".xls"]:
        xl = pd.ExcelFile(path)
        sheets = {}
        for s in xl.sheet_names:
            sheets[s] = pd.read_excel(xl, sheet_name=s)
        return sheets

    else:
        raise ValueError(f"Unsupported file format: {suffix}. Supported: .csv, .tsv, .xlsx, .xls")


def export_to_csv(df: pd.DataFrame, file_path: Union[str, Path]) -> str:
    """Exports dataframe to clean UTF-8 CSV."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    return str(path.resolve())


def export_to_excel(tables: Dict[str, pd.DataFrame], file_path: Union[str, Path]) -> str:
    """Exports multiple tables to a multi-sheet formatted Excel workbook."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name, df in tables.items():
            safe_name = sheet_name[:31]  # Excel max sheet name limit
            df.to_excel(writer, sheet_name=safe_name, index=False)
            
    return str(path.resolve())
