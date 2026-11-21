"""
Virtualized High-Speed Spreadsheet Grid.
Renders 100,000+ row datasets at 60 FPS using QAbstractTableModel.
Supports inline cell editing, column sorting, formula columns, and context menus.
"""

from typing import Any, Optional
import numpy as np
import pandas as pd
from PySide6.QtCore import (
    QAbstractTableModel, QModelIndex, Qt, Signal
)
from PySide6.QtWidgets import (
    QTableView, QMenu, QInputDialog, QMessageBox, QWidget
)


class FastDataFrameModel(QAbstractTableModel):
    """Virtualized DataFrame table model with instant rendering and inline editing."""

    data_modified = Signal()

    def __init__(self, df: Optional[pd.DataFrame] = None):
        super().__init__()
        self._df = df if df is not None else pd.DataFrame()
        self._cached_headers = []
        self._refresh_headers()

    def _refresh_headers(self):
        self._cached_headers = []
        if self._df.empty:
            return
        for col_name, dtype in zip(self._df.columns, self._df.dtypes):
            dt_str = str(dtype).lower()
            if "int" in dt_str or "float" in dt_str:
                prefix = "[NUM] "
            elif "date" in dt_str or "time" in dt_str:
                prefix = "[DATE] "
            else:
                prefix = "[TXT] "
            self._cached_headers.append(f"{prefix}{col_name}")

    def set_dataframe(self, df: pd.DataFrame):
        self.beginResetModel()
        self._df = df.copy() if df is not None else pd.DataFrame()
        self._refresh_headers()
        self.endResetModel()
        self.data_modified.emit()

    def get_dataframe(self) -> pd.DataFrame:
        return self._df

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._df)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._df.columns)

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> Any:
        if not index.isValid() or self._df.empty:
            return None

        row, col = index.row(), index.column()
        val = self._df.iat[row, col]

        if role == Qt.DisplayRole or role == Qt.EditRole:
            if pd.isna(val):
                return ""
            if isinstance(val, (float, np.floating)):
                # Clean currency or percentage formatting heuristic
                col_name = str(self._df.columns[col]).lower()
                if "pct" in col_name or "rate" in col_name or "ratio" in col_name:
                    return f"{val:.1f}%" if abs(val) > 1.0 else f"{val*100:.1f}%"
                if "price" in col_name or "revenue" in col_name or "cost" in col_name or "margin" in col_name or "cf" in col_name:
                    return f"${val:,.2f}"
                return f"{val:.2f}"
            return str(val)

        elif role == Qt.TextAlignmentRole:
            if isinstance(val, (int, float, np.number)):
                return int(Qt.AlignRight | Qt.AlignVCenter)
            return int(Qt.AlignLeft | Qt.AlignVCenter)

        return None

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole) -> Any:
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                if section < len(self._cached_headers):
                    return self._cached_headers[section]
                elif section < len(self._df.columns):
                    return str(self._df.columns[section])
            elif orientation == Qt.Vertical:
                return str(section + 1)
        return None

    def flags(self, index: QModelIndex) -> Qt.ItemFlags:
        if not index.isValid():
            return Qt.NoItemFlags
        return Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsEditable

    def setData(self, index: QModelIndex, value: Any, role: int = Qt.EditRole) -> bool:
        if index.isValid() and role == Qt.EditRole:
            row, col = index.row(), index.column()
            col_name = self._df.columns[col]
            orig_dtype = self._df[col_name].dtype

            try:
                # Coerce data type
                if "int" in str(orig_dtype):
                    coerced_val = int(value)
                elif "float" in str(orig_dtype):
                    clean_str = str(value).replace("$", "").replace("%", "").replace(",", "")
                    coerced_val = float(clean_str)
                else:
                    coerced_val = str(value)

                self._df.iat[row, col] = coerced_val
                self.dataChanged.emit(index, index, [Qt.DisplayRole, Qt.EditRole])
                self.data_modified.emit()
                return True
            except Exception:
                return False
        return False


class SpreadsheetView(QTableView):
    """Production TableView with virtual scrolling, sorting, and context menus."""

    cell_edited_signal = Signal()
    formula_column_requested = Signal(str, str)  # (col_name, formula)
    column_stats_requested = Signal(str)         # (col_name)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.model_df = FastDataFrameModel()
        self.setModel(self.model_df)
        self.setAlternatingRowColors(True)
        self.setSortingEnabled(False)
        self.horizontalHeader().setSectionsClickable(True)
        self.horizontalHeader().sectionClicked.connect(self._on_header_clicked)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._open_context_menu)

        self.model_df.data_modified.connect(self.cell_edited_signal.emit)

    def load_dataframe(self, df: pd.DataFrame):
        self.model_df.set_dataframe(df)
        if df.empty:
            return
        # Fast sample-based column width calculation (O(1) relative to large datasets)
        sample = df.head(30)
        font_metrics = self.fontMetrics()
        for c in range(self.model_df.columnCount()):
            col_name = str(df.columns[c])
            header_w = font_metrics.horizontalAdvance(f"[NUM] {col_name}") + 35
            sample_w = 0
            if not sample.empty:
                val_strs = sample.iloc[:, c].astype(str)
                longest = max(val_strs, key=len) if len(val_strs) > 0 else ""
                sample_w = font_metrics.horizontalAdvance(longest) + 24
            w = max(90, min(280, max(header_w, sample_w)))
            self.setColumnWidth(c, w)

    def get_current_dataframe(self) -> pd.DataFrame:
        return self.model_df.get_dataframe()

    def _on_header_clicked(self, logical_index: int):
        df = self.model_df.get_dataframe()
        if logical_index < len(df.columns):
            col_name = df.columns[logical_index]
            self.column_stats_requested.emit(col_name)

    def _open_context_menu(self, pos):
        index = self.indexAt(pos)
        df = self.model_df.get_dataframe()
        menu = QMenu(self)

        col_name = df.columns[index.column()] if index.isValid() and index.column() < len(df.columns) else None

        if col_name:
            action_sort_asc = menu.addAction(f"Sort '{col_name}' Ascending")
            action_sort_desc = menu.addAction(f"Sort '{col_name}' Descending")
            action_stats = menu.addAction(f"Inspect Column '{col_name}' Statistics")
            menu.addSeparator()

        action_add_formula = menu.addAction("+ Add Calculated Formula Column...")

        action = menu.exec(self.viewport().mapToGlobal(pos))
        if not action:
            return

        if col_name:
            if action == action_sort_asc:
                sorted_df = df.sort_values(by=col_name, ascending=True).reset_index(drop=True)
                self.load_dataframe(sorted_df)
            elif action == action_sort_desc:
                sorted_df = df.sort_values(by=col_name, ascending=False).reset_index(drop=True)
                self.load_dataframe(sorted_df)
            elif action == action_stats:
                self.column_stats_requested.emit(col_name)

        if action == action_add_formula:
            self._prompt_add_formula()

    def _prompt_add_formula(self):
        col_name, ok1 = QInputDialog.getText(self, "New Column", "Enter new column name:")
        if not ok1 or not col_name.strip():
            return
        formula, ok2 = QInputDialog.getText(
            self,
            "Formula Expression",
            f"Enter formula for '{col_name}' (e.g. '=gross_revenue - cogs' or 'effective_price * 1.1'):"
        )
        if ok2 and formula.strip():
            self.formula_column_requested.emit(col_name.strip(), formula.strip())
