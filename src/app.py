"""
Master Application Window: Decision Graph Desktop Engine.
Coordinates the reactive spreadsheet grid, visual node DAG canvas,
1-click decision engines, and details.so editorial styling.
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QTabWidget, QFileDialog, QMessageBox, QInputDialog
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon

from src.ui.style import EDITORIAL_STYLE_QSS
from src.ui.header_nav import HeaderNavToolbar
from src.ui.spreadsheet_view import SpreadsheetView
from src.ui.node_canvas import NodeCanvasView
from src.ui.decision_inspector import DecisionInspectorWidget
from src.ui.chart_inspector import ChartInspectorWidget
from src.ui.swarm_drawer import SwarmDrawerWidget
from src.ui.column_profile import ColumnProfileDialog
from src.engine.dag_engine import DAGEngine
from src.data.synthetic_data import generate_enterprise_suite
from src.data.ingestor import export_to_excel, load_file


class MainWindow(QMainWindow):
    """Production PySide6 Desktop Application Window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Decision Graph - Enterprise Node-Spreadsheet & Analytics Engine")
        self.resize(1380, 880)

        # Apply details.so Light Editorial Theme
        self.setStyleSheet(EDITORIAL_STYLE_QSS)

        # Initialize DAG Engine
        self.dag = DAGEngine()

        self._init_ui()
        self._bootstrap_initial_pipeline()

    def _init_ui(self):
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        root_layout = QVBoxLayout(central_widget)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # 1. Floating Pill Top Navbar
        self.nav_bar = HeaderNavToolbar(self)
        self.nav_bar.action_triggered.connect(self._handle_nav_action)
        root_layout.addWidget(self.nav_bar)

        # 2. Main Workspace Splitter (Horizontal: Canvas/Grid on Left, Inspector on Right)
        main_h_splitter = QSplitter(Qt.Horizontal)

        # Left Work Area (Vertical Splitter: Node Canvas on Top, Spreadsheet Grid on Bottom)
        left_v_splitter = QSplitter(Qt.Vertical)

        # Node Canvas
        self.canvas_view = NodeCanvasView(self)
        self.canvas_view.node_selected.connect(self._on_node_selected)
        self.canvas_view.connection_requested.connect(self._on_connection_created)
        self.canvas_view.node_deleted.connect(self._on_node_deleted)
        left_v_splitter.addWidget(self.canvas_view)

        # Spreadsheet Grid
        self.grid_view = SpreadsheetView(self)
        self.grid_view.formula_column_requested.connect(self._on_formula_column_requested)
        self.grid_view.column_stats_requested.connect(self._on_column_stats_requested)
        self.grid_view.cell_edited_signal.connect(self._on_grid_cell_edited)
        left_v_splitter.addWidget(self.grid_view)

        # Set 45% / 55% vertical distribution
        left_v_splitter.setSizes([380, 440])
        main_h_splitter.addWidget(left_v_splitter)

        # Right Side Panel: Tabbed Inspector (Decisions, Charts, Swarm, Audit Log)
        self.side_tabs = QTabWidget()

        # Tab 1: Executive Decision Inspector
        self.decision_inspector = DecisionInspectorWidget(self)
        self.decision_inspector.action_applied.connect(self._on_decision_action_applied)
        self.side_tabs.addTab(self.decision_inspector, "DECISION COCKPIT")

        # Tab 2: Visual Distribution Charts
        self.chart_inspector = ChartInspectorWidget(self)
        self.side_tabs.addTab(self.chart_inspector, "VISUAL ANALYTICS")

        # Tab 3: Swarm Co-Pilot
        self.swarm_drawer = SwarmDrawerWidget(self)
        self.swarm_drawer.swarm_action_triggered.connect(self._on_swarm_action)
        self.side_tabs.addTab(self.swarm_drawer, "SWARM CO-PILOT")

        # Tab 4: Audit Ledger
        self.audit_table = SpreadsheetView(self)
        self.side_tabs.addTab(self.audit_table, "AUDIT LEDGER")

        main_h_splitter.addWidget(self.side_tabs)

        # Set 70% / 30% horizontal distribution
        main_h_splitter.setSizes([950, 410])
        root_layout.addWidget(main_h_splitter)

    def _bootstrap_initial_pipeline(self):
        """Creates a verified starter pipeline so the application is ready on launch."""
        # 1. Source Node: Synthetic Enterprise Dataset
        n1 = self.dag.add_node("source_synthetic", "Enterprise Data Source", params={"table": "transactions"}, x=40, y=80)
        self.canvas_view.add_node_item(n1.node_id, n1.node_type, n1.title, n1.x, n1.y)

        # 2. Transform Node: Clean Filter
        n2 = self.dag.add_node("transform_filter", "Margin Quality Filter", params={"query": "gross_margin >= 0"}, x=320, y=80)
        self.canvas_view.add_node_item(n2.node_id, n2.node_type, n2.title, n2.x, n2.y)
        self.dag.add_edge(n1.node_id, n2.node_id)
        self.canvas_view.add_connection_item(n1.node_id, n2.node_id)

        # 3. Analytics Node: Causal Elasticity
        n3 = self.dag.add_node("analytics_causal_elasticity", "Causal Price Elasticity", params={}, x=600, y=80)
        self.canvas_view.add_node_item(n3.node_id, n3.node_type, n3.title, n3.x, n3.y)
        self.dag.add_edge(n2.node_id, n3.node_id)
        self.canvas_view.add_connection_item(n2.node_id, n3.node_id)

        # Execute Pipeline
        self.dag.execute_all()
        for node_id in self.dag.nodes:
            self.canvas_view.update_node_status(node_id, self.dag.nodes[node_id].status)

        # Load n1 data into spreadsheet
        self._on_node_selected(n1.node_id)

    def _on_node_selected(self, node_id: str):
        node = self.dag.nodes.get(node_id)
        if not node:
            return

        # 1. Update Spreadsheet View
        if hasattr(node.output_data, "columns"):  # Is DataFrame
            self.grid_view.load_dataframe(node.output_data)
            self.swarm_drawer.audit_dataset(node.output_data)
        elif isinstance(node.output_data, dict):
            # If dictionary contains summary dataframe, load it
            summary_df = next((v for v in node.output_data.values() if hasattr(v, "columns")), None)
            if summary_df is not None:
                self.grid_view.load_dataframe(summary_df)
            self.decision_inspector.update_decision(node.node_type.replace("analytics_", ""), node.output_data)
            self._update_charts_from_dict(node.node_type, node.output_data)

        # 2. Update Audit Ledger Tab
        self.audit_table.load_dataframe(self.dag.audit_ledger.to_dataframe())

    def _update_charts_from_dict(self, node_type: str, data: Dict[str, Any]):
        try:
            if "elasticity" in node_type:
                self.chart_inspector.plot_elasticity(data)
                self.side_tabs.setCurrentWidget(self.chart_inspector)
            elif "cfar" in node_type:
                self.chart_inspector.plot_cfar(data)
                self.side_tabs.setCurrentWidget(self.chart_inspector)
            elif "anomaly" in node_type:
                df = self.grid_view.get_current_dataframe()
                anomalies_df = data.get("anomalies_df")
                if anomalies_df is not None:
                    self.chart_inspector.plot_anomalies(df, anomalies_df)
                    self.side_tabs.setCurrentWidget(self.chart_inspector)
            elif "inventory" in node_type or "meio" in node_type:
                self.chart_inspector.plot_inventory(data)
                self.side_tabs.setCurrentWidget(self.chart_inspector)
            elif "cohort" in node_type or "churn" in node_type:
                self.chart_inspector.plot_cohort_churn(data)
                self.side_tabs.setCurrentWidget(self.chart_inspector)
            elif "attribution" in node_type or "markov" in node_type:
                self.chart_inspector.plot_attribution(data)
                self.side_tabs.setCurrentWidget(self.chart_inspector)
            elif "network" in node_type or "fragility" in node_type:
                self.chart_inspector.plot_network(data)
                self.side_tabs.setCurrentWidget(self.chart_inspector)
            elif "bandit" in node_type:
                self.chart_inspector.plot_bandit(data)
                self.side_tabs.setCurrentWidget(self.chart_inspector)
            elif "knapsack" in node_type:
                self.chart_inspector.plot_knapsack(data)
                self.side_tabs.setCurrentWidget(self.chart_inspector)
        except Exception as e:
            print(f"[Chart Error] {e}")

    def _on_connection_created(self, source_id: str, target_id: str):
        if self.dag.add_edge(source_id, target_id):
            self.canvas_view.add_connection_item(source_id, target_id)
            self.dag.execute_all()
            for n_id, n in self.dag.nodes.items():
                self.canvas_view.update_node_status(n_id, n.status)
            self._on_node_selected(target_id)
        else:
            QMessageBox.warning(self, "Cycle Detected", "Cannot create connection: Cycle detected in pipeline DAG.")

    def _on_node_deleted(self, node_id: str):
        self.dag.remove_node(node_id)

    def _on_grid_cell_edited(self):
        # Cell was edited inline in the spreadsheet: re-run pipeline downstream
        curr_df = self.grid_view.get_current_dataframe()
        # Find active node
        for n_id, n in self.dag.nodes.items():
            if n.output_data is not None and hasattr(n.output_data, "shape") and n.output_data.shape == curr_df.shape:
                n.output_data = curr_df
                break
        self.dag.execute_all()
        self.audit_table.load_dataframe(self.dag.audit_ledger.to_dataframe())

    def _on_formula_column_requested(self, col_name: str, formula: str):
        df = self.grid_view.get_current_dataframe()
        node = self.dag.add_node("transform_formula", f"Formula: {col_name}", params={"new_column": col_name, "formula": formula}, x=350, y=200)
        self.canvas_view.add_node_item(node.node_id, node.node_type, node.title, node.x, node.y)
        
        # Connect to first available source
        src_id = list(self.dag.nodes.keys())[0]
        self._on_connection_created(src_id, node.node_id)

    def _on_column_stats_requested(self, col_name: str):
        df = self.grid_view.get_current_dataframe()
        if col_name in df.columns:
            dlg = ColumnProfileDialog(col_name, df[col_name], self)
            dlg.exec()

    def _handle_nav_action(self, action_id: str):
        df = self.grid_view.get_current_dataframe()

        if action_id == "run_anomaly":
            node = self.dag.add_node("analytics_anomaly_sweep", "Anomaly & Margin Leak Sweep", x=620, y=180)
            self.canvas_view.add_node_item(node.node_id, node.node_type, node.title, node.x, node.y)
            src_id = list(self.dag.nodes.keys())[0]
            self._on_connection_created(src_id, node.node_id)

        elif action_id == "run_elasticity":
            node = self.dag.add_node("analytics_causal_elasticity", "Price Elasticity Engine", x=620, y=80)
            self.canvas_view.add_node_item(node.node_id, node.node_type, node.title, node.x, node.y)
            src_id = list(self.dag.nodes.keys())[0]
            self._on_connection_created(src_id, node.node_id)

        elif action_id == "run_cfar":
            node = self.dag.add_node("analytics_cfar", "Cash Flow at Risk (5,000 Paths)", params={"covenant_floor": 300000.0}, x=620, y=280)
            self.canvas_view.add_node_item(node.node_id, node.node_type, node.title, node.x, node.y)
            src_id = list(self.dag.nodes.keys())[0]
            self._on_connection_created(src_id, node.node_id)

        elif action_id == "run_inventory":
            inv_src = next((n for n in self.dag.nodes.values() if n.node_type == "source_synthetic" and n.params.get("table") == "inventory"), None)
            if not inv_src:
                inv_src = self.dag.add_node("source_synthetic", "Inventory Table Source", params={"table": "inventory"}, x=40, y=260)
                self.canvas_view.add_node_item(inv_src.node_id, inv_src.node_type, inv_src.title, inv_src.x, inv_src.y)
                self.dag.execute_node(inv_src.node_id)
            node = self.dag.add_node("analytics_inventory_meio", "Multi-Echelon Inventory MEIO", x=420, y=260)
            self.canvas_view.add_node_item(node.node_id, node.node_type, node.title, node.x, node.y)
            self._on_connection_created(inv_src.node_id, node.node_id)

        elif action_id == "run_network":
            src_id = list(self.dag.nodes.keys())[0]
            node = self.dag.add_node("analytics_network_fragility", "Network Fragility Graph", x=620, y=360)
            self.canvas_view.add_node_item(node.node_id, node.node_type, node.title, node.x, node.y)
            self._on_connection_created(src_id, node.node_id)

        elif action_id == "toggle_swarm":
            self.side_tabs.setCurrentWidget(self.swarm_drawer)
            self.swarm_drawer.audit_dataset(df)

        elif action_id == "export_excel":
            file_path, _ = QFileDialog.getSaveFileName(self, "Export Formatted Workbook", "Decision_Summary.xlsx", "Excel Files (*.xlsx)")
            if file_path:
                tables = {"Current_View": df, "Audit_Ledger": self.dag.audit_ledger.to_dataframe()}
                saved_path = export_to_excel(tables, file_path)
                QMessageBox.information(self, "Export Successful", f"Workbook exported cleanly to:\n{saved_path}")

        elif action_id == "add_node_dialog":
            self._prompt_add_node()

    def _prompt_add_node(self):
        types = [
            "transform_filter (Filter Rows)",
            "transform_formula (Add Formula)",
            "transform_dedup (Deduplicate)",
            "transform_groupby (Group & Aggregate)",
            "analytics_causal_elasticity (Price Elasticity)",
            "analytics_cfar (Cash Flow at Risk)",
            "analytics_anomaly_sweep (Anomaly Sweep)",
            "analytics_inventory_meio (Stock MEIO)",
            "analytics_cohort_churn (Cohort Churn)",
            "analytics_markov_attribution (Markov Attribution)",
            "analytics_mixed_logit (Cannibalization)",
            "analytics_bandit (Bandit Lab)",
            "analytics_network_fragility (Network Fragility)",
            "analytics_knapsack (Capital Allocation)",
        ]
        choice, ok = QInputDialog.getItem(self, "Add Node", "Select Node Type to Add:", types, 0, False)
        if ok and choice:
            clean_type = choice.split()[0]
            node = self.dag.add_node(clean_type, choice.split("(")[1].replace(")", ""), x=400, y=250)
            self.canvas_view.add_node_item(node.node_id, node.node_type, node.title, node.x, node.y)

    def _on_swarm_action(self, action_type: str, payload: dict):
        if action_type == "filter_margin_leaks":
            node = self.dag.add_node("transform_filter", "Quarantine Negative Margins", params={"query": payload.get("query", "gross_margin >= 0")}, x=350, y=150)
            self.canvas_view.add_node_item(node.node_id, node.node_type, node.title, node.x, node.y)
            src_id = list(self.dag.nodes.keys())[0]
            self._on_connection_created(src_id, node.node_id)
            QMessageBox.information(self, "Swarm Action Applied", "Quarantine filter node attached to pipeline. Cleaned rows displayed below.")
        elif action_type == "trigger_elasticity":
            self._handle_nav_action("run_elasticity")

    def _on_decision_action_applied(self, action_desc: str):
        self.dag.audit_ledger.record_action(
            node_id="DECISION_COCKPIT",
            action_type="EXECUTIVE_OVERRIDE",
            rationale=action_desc,
        )
        self.audit_table.load_dataframe(self.dag.audit_ledger.to_dataframe())
        QMessageBox.information(self, "Action Executed", f"Action logged to tamper-evident audit ledger:\n\n{action_desc}")
