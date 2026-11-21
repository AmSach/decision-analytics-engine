"""
Formula Parser and Evaluator.
Evaluates computed column expressions across pandas DataFrame columns
supporting both Excel-style formulas (=A * B) and standard mathematical expressions.
"""

import re
from typing import Tuple, Optional
import numpy as np
import pandas as pd


def evaluate_formula(df: pd.DataFrame, formula: str, new_col_name: str) -> Tuple[Optional[pd.Series], Optional[str]]:
    """
    Evaluates a formula against the dataframe:
    Examples:
    - "=gross_revenue - cogs"
    - "gross_margin / gross_revenue * 100"
    - "=ROUND(effective_price * 1.1, 2)"
    - "quantity * 2"
    Returns: (computed_series, error_message)
    """
    clean_expr = formula.strip()
    if clean_expr.startswith("="):
        clean_expr = clean_expr[1:].strip()

    if not clean_expr:
        return None, "Empty formula expression."

    # Replace Excel functions with NumPy equivalents
    clean_expr = re.sub(r'\bROUND\s*\(', 'np.round(', clean_expr, flags=re.IGNORECASE)
    clean_expr = re.sub(r'\bABS\s*\(', 'np.abs(', clean_expr, flags=re.IGNORECASE)
    clean_expr = re.sub(r'\bLOG\s*\(', 'np.log(', clean_expr, flags=re.IGNORECASE)
    clean_expr = re.sub(r'\bSQRT\s*\(', 'np.sqrt(', clean_expr, flags=re.IGNORECASE)
    clean_expr = re.sub(r'\bMAX\s*\(', 'np.maximum(', clean_expr, flags=re.IGNORECASE)
    clean_expr = re.sub(r'\bMIN\s*\(', 'np.minimum(', clean_expr, flags=re.IGNORECASE)

    # Build evaluation context with dataframe columns as local variables
    local_env = {col: df[col] for col in df.columns}
    local_env["np"] = np
    local_env["pd"] = pd

    # Safe built-ins
    safe_builtins = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "len": len,
        "int": int,
        "float": float,
    }

    try:
        result = eval(clean_expr, {"__builtins__": safe_builtins}, local_env)
        if isinstance(result, (int, float, np.number)):
            result = pd.Series([result] * len(df), index=df.index)
        elif not isinstance(result, pd.Series):
            result = pd.Series(result, index=df.index)
        return result, None
    except Exception as e:
        return None, f"Formula Evaluation Error: {str(e)}"
