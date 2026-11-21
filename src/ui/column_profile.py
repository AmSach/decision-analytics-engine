"""
Column Data Profiler and Summary Statistics Dialog.
Displays distribution statistics, null counts, and categorical frequency bars.
"""

from typing import Optional
import numpy as np
import pandas as pd
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QPushButton, QGridLayout
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont


class ColumnProfileDialog(QDialog):
    """Clean editorial dialog profiling column metrics and distributions."""

    def __init__(self, col_name: str, series: pd.Series, parent: Optional[QDialog] = None):
        super().__init__(parent)
        self.setWindowTitle(f"Column Profile: {col_name}")
        self.resize(380, 420)
        self._init_ui(col_name, series)

    def _init_ui(self, col_name: str, s: pd.Series):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        # Header
        title = QLabel(f"Profile: {col_name}")
        title.setFont(QFont("-apple-system, sans-serif", 14, QFont.Bold))
        title.setStyleSheet("color: #09090b;")
        layout.addWidget(title)

        # Data type and nulls row
        meta_frame = QFrame()
        meta_frame.setStyleSheet("background-color: #f4f4f5; border-radius: 8px; padding: 10px;")
        meta_layout = QGridLayout(meta_frame)
        meta_layout.setContentsMargins(8, 8, 8, 8)

        total_rows = len(s)
        null_count = s.isna().sum()
        valid_count = total_rows - null_count
        valid_pct = (valid_count / max(1, total_rows)) * 100.0

        meta_layout.addWidget(QLabel("<b>Data Type:</b>"), 0, 0)
        meta_layout.addWidget(QLabel(str(s.dtype)), 0, 1)
        meta_layout.addWidget(QLabel("<b>Total Rows:</b>"), 1, 0)
        meta_layout.addWidget(QLabel(f"{total_rows:,}"), 1, 1)
        meta_layout.addWidget(QLabel("<b>Valid Cells:</b>"), 2, 0)
        meta_layout.addWidget(QLabel(f"{valid_count:,} ({valid_pct:.1f}%)"), 2, 1)
        meta_layout.addWidget(QLabel("<b>Null Count:</b>"), 3, 0)
        meta_layout.addWidget(QLabel(f"{null_count:,}"), 3, 1)

        layout.addWidget(meta_frame)

        # Statistics Section
        stats_frame = QFrame()
        stats_frame.setStyleSheet("background-color: #ffffff; border: 1px solid #e4e4e7; border-radius: 8px;")
        stats_layout = QGridLayout(stats_frame)
        stats_layout.setContentsMargins(12, 12, 12, 12)
        stats_layout.setSpacing(8)

        if np.issubdtype(s.dtype, np.number):
            valid_num = s.dropna()
            mean_val = valid_num.mean()
            median_val = valid_num.median()
            std_val = valid_num.std()
            min_val = valid_num.min()
            max_val = valid_num.max()

            stats_layout.addWidget(QLabel("Mean:"), 0, 0)
            stats_layout.addWidget(QLabel(f"<b>{mean_val:,.2f}</b>"), 0, 1)
            stats_layout.addWidget(QLabel("Median:"), 1, 0)
            stats_layout.addWidget(QLabel(f"<b>{median_val:,.2f}</b>"), 1, 1)
            stats_layout.addWidget(QLabel("Std Dev:"), 2, 0)
            stats_layout.addWidget(QLabel(f"<b>{std_val:,.2f}</b>"), 2, 1)
            stats_layout.addWidget(QLabel("Minimum:"), 3, 0)
            stats_layout.addWidget(QLabel(f"<b>{min_val:,.2f}</b>"), 3, 1)
            stats_layout.addWidget(QLabel("Maximum:"), 4, 0)
            stats_layout.addWidget(QLabel(f"<b>{max_val:,.2f}</b>"), 4, 1)
        else:
            # Categorical statistics
            n_unique = s.nunique()
            top_vals = s.value_counts().head(5)

            stats_layout.addWidget(QLabel("Unique Values:"), 0, 0)
            stats_layout.addWidget(QLabel(f"<b>{n_unique:,}</b>"), 0, 1)
            
            row_idx = 1
            for val, count in top_vals.items():
                pct = (count / max(1, total_rows)) * 100.0
                stats_layout.addWidget(QLabel(f"'{val}':"), row_idx, 0)
                stats_layout.addWidget(QLabel(f"{count:,} ({pct:.1f}%)"), row_idx, 1)
                row_idx += 1

        layout.addWidget(stats_frame)
        layout.addStretch()

        close_btn = QPushButton("Close")
        close_btn.setObjectName("primaryActionBtn")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)
