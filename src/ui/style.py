"""
Institutional Light-Mode Editorial Stylesheet.
Fused with FestOS control-tower clarity and details.so craftsmanship.
High contrast, zero emojis, zero CSS parse errors, crisp typography.
"""

EDITORIAL_STYLE_QSS = """
/* Global Application Palette & Font Reset */
QWidget {
    color: #0f172a;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Geist", "Inter", sans-serif;
    font-size: 12.5px;
    selection-background-color: #3b82f6;
    selection-color: #ffffff;
}

/* Ensure labels don't get ugly background blocks */
QLabel {
    background: transparent;
}

/* Main Window & Containers */
QMainWindow {
    background-color: #f1f5f9;
}

QFrame#cardFrame {
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
}

/* Top Institutional Toolbar */
QFrame#headerNav {
    background-color: #ffffff;
    border-bottom: 1px solid #e2e8f0;
    padding: 6px 16px;
    margin: 0px;
}

/* High-Impact Institutional Action Buttons */
QPushButton {
    background-color: #ffffff;
    color: #1e293b;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    padding: 5px 12px;
    font-weight: 600;
    font-size: 11px;
    letter-spacing: 0.3px;
}

QPushButton:hover {
    background-color: #f8fafc;
    border-color: #94a3b8;
    color: #0f172a;
}

QPushButton:pressed {
    background-color: #e2e8f0;
    border-color: #64748b;
}

QPushButton#primaryActionBtn {
    background-color: #0f172a;
    color: #ffffff;
    border: 1px solid #0f172a;
}

QPushButton#primaryActionBtn:hover {
    background-color: #1e293b;
    border-color: #1e293b;
}

QPushButton#primaryActionBtn:pressed {
    background-color: #334155;
}

QPushButton#emeraldActionBtn {
    background-color: #ecfdf5;
    color: #047857;
    border: 1px solid #a7f3d0;
}

QPushButton#emeraldActionBtn:hover {
    background-color: #d1fae5;
    border-color: #6ee7b7;
}

QPushButton#indigoActionBtn {
    background-color: #eef2ff;
    color: #4338ca;
    border: 1px solid #c7d2fe;
}

QPushButton#indigoActionBtn:hover {
    background-color: #e0e7ff;
    border-color: #a5b4fc;
}

/* Info Tooltip Hover Badge (i) */
QPushButton#infoHoverBtn {
    background-color: #f1f5f9;
    color: #475569;
    border: 1px solid #cbd5e1;
    border-radius: 9px;
    font-family: "JetBrains Mono", Consolas, monospace;
    font-size: 10px;
    font-weight: 700;
    padding: 0px;
    min-width: 18px;
    max-width: 18px;
    min-height: 18px;
    max-height: 18px;
}

QPushButton#infoHoverBtn:hover {
    background-color: #0f172a;
    color: #ffffff;
    border-color: #0f172a;
}

/* Virtualized Spreadsheet QTableView */
QTableView {
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    gridline-color: #f1f5f9;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    font-size: 12px;
    alternate-background-color: #f8fafc;
}

QTableView::item {
    padding: 4px 8px;
    border-bottom: 1px solid #f1f5f9;
}

QTableView::item:selected {
    background-color: #e0e7ff;
    color: #1e1b4b;
}

QHeaderView::section {
    background-color: #f8fafc;
    color: #475569;
    border: none;
    border-right: 1px solid #e2e8f0;
    border-bottom: 1px solid #cbd5e1;
    padding: 6px 10px;
    font-weight: 700;
    font-size: 10.5px;
    letter-spacing: 0.5px;
}

QHeaderView::section:hover {
    background-color: #f1f5f9;
    color: #0f172a;
}

/* Scrollbars */
QScrollBar:vertical {
    border: none;
    background: #f1f5f9;
    width: 8px;
    border-radius: 4px;
}

QScrollBar::handle:vertical {
    background: #cbd5e1;
    min-height: 25px;
    border-radius: 4px;
}

QScrollBar::handle:vertical:hover {
    background: #94a3b8;
}

QScrollBar:horizontal {
    border: none;
    background: #f1f5f9;
    height: 8px;
    border-radius: 4px;
}

QScrollBar::handle:horizontal {
    background: #cbd5e1;
    min-width: 25px;
    border-radius: 4px;
}

/* Tabs & Splitters */
QTabWidget::pane {
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    background: #ffffff;
}

QTabBar::tab {
    background: #f1f5f9;
    color: #64748b;
    border: 1px solid #e2e8f0;
    border-bottom: none;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    padding: 6px 14px;
    margin-right: 2px;
    font-weight: 600;
    font-size: 11px;
    letter-spacing: 0.4px;
}

QTabBar::tab:selected {
    background: #ffffff;
    color: #0f172a;
    border-bottom: 2px solid #3b82f6;
}

QSplitter::handle {
    background-color: #e2e8f0;
}

QSplitter::handle:horizontal {
    width: 2px;
}

QSplitter::handle:vertical {
    height: 2px;
}

/* Status Chips & Badges */
QLabel#statusPillGreen {
    background-color: #dcfce7;
    color: #15803d;
    border: 1px solid #86efac;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.5px;
}

QLabel#brandTitle {
    font-size: 14px;
    font-weight: 800;
    color: #0f172a;
    letter-spacing: -0.2px;
}
"""
