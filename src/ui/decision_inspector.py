"""
Actionable Executive Decision Card & Inspector.
Transforms complex econometric and machine learning outputs into
crystal-clear, board-level decisions with the (i) info hover card.
"""

from typing import Dict, Any, Optional
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QPushButton,
    QScrollArea, QGridLayout
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont

from src.ui.info_tooltip import InfoHoverBadge


class DecisionInspectorWidget(QWidget):
    """
    Renders the executive decision matrix and downside stress test
    for the selected algorithm or pipeline node.
    """

    action_applied = Signal(str)  # action description

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # Header with Title and Info Badge
        header_row = QHBoxLayout()
        self.title_label = QLabel("Executive Decision Cockpit")
        self.title_label.setFont(QFont("-apple-system, sans-serif", 13, QFont.Bold))
        self.title_label.setStyleSheet("color: #09090b;")
        header_row.addWidget(self.title_label)

        self.info_badge = InfoHoverBadge("causal_elasticity", self)
        header_row.addWidget(self.info_badge)
        header_row.addStretch()

        layout.addLayout(header_row)

        # Main Actionable Recommendation Callout Box
        self.action_card = QFrame()
        self.action_card.setObjectName("cardFrame")
        self.action_card.setStyleSheet("""
            QFrame#cardFrame {
                background-color: #ecfdf5;
                border: 1px solid #a7f3d0;
                border-radius: 10px;
                padding: 10px;
            }
        """)
        action_card_layout = QVBoxLayout(self.action_card)
        action_card_layout.setContentsMargins(10, 10, 10, 10)
        action_card_layout.setSpacing(6)

        action_header = QLabel("RECOMMENDED ACTION")
        action_header.setFont(QFont("-apple-system, sans-serif", 9, QFont.Bold))
        action_header.setStyleSheet("color: #065f46; letter-spacing: 0.5px;")
        action_card_layout.addWidget(action_header)

        self.action_text = QLabel("Select an analytics node or trigger a 1-click decision action to view recommendations.")
        self.action_text.setFont(QFont("-apple-system, sans-serif", 11, QFont.DemiBold))
        self.action_text.setStyleSheet("color: #064e3b; line-height: 1.4;")
        self.action_text.setWordWrap(True)
        action_card_layout.addWidget(self.action_text)

        layout.addWidget(self.action_card)

        # Decision Metrics Grid
        self.metrics_container = QWidget()
        self.metrics_grid = QGridLayout(self.metrics_container)
        self.metrics_grid.setContentsMargins(0, 4, 0, 4)
        self.metrics_grid.setSpacing(8)
        layout.addWidget(self.metrics_container)

        # Downside Stress Test Card
        self.stress_card = QFrame()
        self.stress_card.setObjectName("cardFrame")
        self.stress_card.setStyleSheet("""
            QFrame#cardFrame {
                background-color: #f8fafc;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                padding: 8px;
            }
        """)
        stress_layout = QVBoxLayout(self.stress_card)
        stress_layout.setContentsMargins(8, 8, 8, 8)
        stress_layout.setSpacing(4)

        stress_header = QLabel("DOWNSIDE SENSITIVITY (RECOMMENDATION HOLDS)")
        stress_header.setFont(QFont("-apple-system, sans-serif", 8, QFont.Bold))
        stress_header.setStyleSheet("color: #475569;")
        stress_layout.addWidget(stress_header)

        self.stress_text = QLabel("Sensitivity bounds will populate upon model execution.")
        self.stress_text.setFont(QFont("-apple-system, sans-serif", 10))
        self.stress_text.setStyleSheet("color: #334155;")
        self.stress_text.setWordWrap(True)
        stress_layout.addWidget(self.stress_text)

        layout.addWidget(self.stress_card)

        # Apply Action Button
        self.apply_btn = QPushButton("Execute Recommended Action")
        self.apply_btn.setObjectName("primaryActionBtn")
        self.apply_btn.clicked.connect(self._on_apply_clicked)
        layout.addWidget(self.apply_btn)

        layout.addStretch()

    def update_decision(self, model_key: str, results: Dict[str, Any]):
        """Populates the decision cockpit with model results."""
        self.info_badge.tooltip_key = model_key
        self.info_badge.setToolTip(self.info_badge.toolTip())  # Re-trigger update

        title = results.get("title") or model_key.replace("_", " ").title()
        self.title_label.setText(title)

        action = results.get("decision_action", "No explicit action recommended.")
        self.action_text.setText(action)

        stress = results.get("downside_stress_test", "Sensitivity verified across base and downside cases.")
        self.stress_text.setText(stress)

        # Clear old metrics
        for i in reversed(range(self.metrics_grid.count())):
            widget = self.metrics_grid.itemAt(i).widget()
            if widget:
                widget.setParent(None)

        # Render top 4 metric badges
        metrics = []
        for k, v in results.items():
            if k not in ["decision_action", "downside_stress_test", "scenarios", "policy_df", "anomalies_df", "summary_df", "retention_matrix"]:
                if isinstance(v, (int, float, str)):
                    metrics.append((k.replace("_", " ").title(), str(v)))

        for idx, (label, val) in enumerate(metrics[:4]):
            row = idx // 2
            col = idx % 2
            card = self._create_metric_card(label, val)
            self.metrics_grid.addWidget(card, row, col)

    def _create_metric_card(self, label: str, val: str) -> QWidget:
        box = QFrame()
        box.setStyleSheet("""
            background-color: #ffffff;
            border: 1px solid #e4e4e7;
            border-radius: 6px;
            padding: 6px;
        """)
        lay = QVBoxLayout(box)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(2)

        lbl = QLabel(label)
        lbl.setFont(QFont("-apple-system, sans-serif", 8, QFont.Medium))
        lbl.setStyleSheet("color: #71717a;")
        lay.addWidget(lbl)

        val_lbl = QLabel(val)
        val_lbl.setFont(QFont("JetBrains Mono, monospace", 11, QFont.Bold))
        val_lbl.setStyleSheet("color: #09090b;")
        lay.addWidget(val_lbl)

        return box

    def _on_apply_clicked(self):
        self.action_applied.emit(self.action_text.text())
