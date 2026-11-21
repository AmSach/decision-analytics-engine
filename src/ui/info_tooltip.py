"""
Interactive (i) Information Hover Component.
Renders a polished circular badge that presents an accurate,
executive-ready model explanation card on hover.
"""

from PySide6.QtWidgets import (
    QPushButton, QWidget, QVBoxLayout, QLabel, QFrame, QToolTip
)
from PySide6.QtCore import Qt, QPoint
from PySide6.QtGui import QCursor

from src.tooltips import MODEL_TOOLTIPS, NODE_TOOLTIPS


def get_formatted_tooltip_html(key: str) -> str:
    """Formats a rich HTML card for the given model or node key."""
    info = MODEL_TOOLTIPS.get(key) or NODE_TOOLTIPS.get(key)
    if not info:
        return f"<b>{key}</b><br/>No detailed tooltip available."

    title = info.get("title", key)
    plain = info.get("plain_english", "")
    formula = info.get("formula", "")
    rule = info.get("decision_rule", "")
    example = info.get("example", "")

    html = f"""
    <div style='background-color: #ffffff; color: #09090b; padding: 12px; border: 1px solid #e4e4e7; border-radius: 8px; width: 340px; font-family: -apple-system, sans-serif;'>
        <div style='font-size: 13px; font-weight: 700; color: #09090b; margin-bottom: 6px; border-bottom: 1px solid #f4f4f5; padding-bottom: 4px;'>
            {title}
        </div>
        <div style='font-size: 12px; color: #3f3f46; line-height: 1.4; margin-bottom: 8px;'>
            {plain}
        </div>
        <div style='background-color: #f4f4f5; padding: 6px 8px; border-radius: 6px; font-family: "JetBrains Mono", Consolas, monospace; font-size: 10px; color: #18181b; margin-bottom: 8px;'>
            <b>Math:</b> {formula}
        </div>
        <div style='font-size: 11px; color: #047857; background-color: #ecfdf5; padding: 6px 8px; border-radius: 6px; border: 1px solid #a7f3d0; margin-bottom: 6px;'>
            <b>Decision Rule:</b> {rule}
        </div>
        <div style='font-size: 10px; color: #71717a; font-style: italic;'>
            <b>Example:</b> {example}
        </div>
    </div>
    """
    return html.strip()


class InfoHoverBadge(QPushButton):
    """
    Circular (i) badge with rich HTML hover card.
    """

    def __init__(self, tooltip_key: str, parent: QWidget = None):
        super().__init__("i", parent)
        self.tooltip_key = tooltip_key
        self.setObjectName("infoHoverBtn")
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip(get_formatted_tooltip_html(tooltip_key))
