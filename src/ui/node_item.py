"""
Custom QGraphicsItem for Canvas Nodes and Ports.
Renders tactile, rounded node cards with input/output ports,
status pills, and (i) info badges.
"""

from typing import Any, Optional, List
from PySide6.QtWidgets import (
    QGraphicsItem, QGraphicsRectItem, QGraphicsTextItem, QGraphicsDropShadowEffect,
    QStyleOptionGraphicsItem, QWidget, QToolTip
)
from PySide6.QtCore import Qt, QRectF, QPointF
from PySide6.QtGui import (
    QBrush, QPen, QColor, QFont, QPainter, QPainterPath
)

from src.ui.info_tooltip import get_formatted_tooltip_html


class PortItem(QGraphicsItem):
    """Circular connection port (Input or Output) on a node."""

    def __init__(self, is_input: bool, parent_node: 'NodeCardItem'):
        super().__init__(parent_node)
        self.is_input = is_input
        self.parent_node = parent_node
        self.radius = 6.0
        self.setAcceptHoverEvents(True)
        self.is_hovered = False

    def boundingRect(self) -> QRectF:
        r = self.radius + 3
        return QRectF(-r, -r, 2 * r, 2 * r)

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: Optional[QWidget] = None):
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Color based on port state
        if self.is_hovered:
            brush = QBrush(QColor("#10b981"))  # Emerald highlight
            pen = QPen(QColor("#065f46"), 2)
        else:
            brush = QBrush(QColor("#ffffff"))
            pen = QPen(QColor("#71717a"), 2)

        painter.setBrush(brush)
        painter.setPen(pen)
        painter.drawEllipse(QPointF(0, 0), self.radius, self.radius)

    def hoverEnterEvent(self, event):
        self.is_hovered = True
        self.update()
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self.is_hovered = False
        self.update()
        super().hoverLeaveEvent(event)

    def get_scene_pos(self) -> QPointF:
        return self.mapToScene(QPointF(0, 0))


class NodeCardItem(QGraphicsItem):
    """Tactile Node Card Item on the QGraphicsScene."""

    def __init__(self, node_id: str, node_type: str, title: str, x: float = 100.0, y: float = 100.0):
        super().__init__()
        self.node_id = node_id
        self.node_type = node_type
        self.title = title
        self.status = "PENDING"
        self.width = 200.0
        self.height = 80.0

        self.setPos(x, y)
        self.setFlags(
            QGraphicsItem.ItemIsMovable |
            QGraphicsItem.ItemIsSelectable |
            QGraphicsItem.ItemSendsGeometryChanges
        )
        self.setAcceptHoverEvents(True)

        # Input Port on Left Center
        self.input_port = PortItem(is_input=True, parent_node=self)
        self.input_port.setPos(0, self.height / 2.0)

        # Output Port on Right Center
        self.output_port = PortItem(is_input=False, parent_node=self)
        self.output_port.setPos(self.width, self.height / 2.0)

        # Tooltip HTML
        self.setToolTip(get_formatted_tooltip_html(node_type))

    def boundingRect(self) -> QRectF:
        return QRectF(-10, -10, self.width + 20, self.height + 20)

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: Optional[QWidget] = None):
        painter.setRenderHint(QPainter.Antialiasing)

        # Rounded rectangle path
        rect = QRectF(0, 0, self.width, self.height)
        path = QPainterPath()
        path.addRoundedRect(rect, 10, 10)

        # Card Fill & Border
        if self.isSelected():
            border_color = QColor("#10b981")  # Emerald selection border
            border_width = 2.0
            bg_color = QColor("#ffffff")
        else:
            border_color = QColor("#e4e4e7")  # 1px subtle neutral border
            border_width = 1.0
            bg_color = QColor("#ffffff")

        painter.fillPath(path, QBrush(bg_color))
        painter.setPen(QPen(border_color, border_width))
        painter.drawPath(path)

        # Top Header Accent Pill
        header_rect = QRectF(0, 0, self.width, 24)
        header_path = QPainterPath()
        header_path.addRoundedRect(header_rect, 10, 10)
        # Clip header
        painter.save()
        painter.setClipPath(path)
        painter.fillPath(header_rect, QBrush(QColor("#f4f4f5")))
        painter.setPen(QPen(QColor("#e4e4e7"), 1))
        painter.drawLine(0, 24, self.width, 24)
        painter.restore()

        # Category Badge in Header
        category_text = self.node_type.split("_")[0].upper()
        painter.setFont(QFont("-apple-system, sans-serif", 8, QFont.Bold))
        painter.setPen(QColor("#71717a"))
        painter.drawText(10, 16, category_text)

        # Small (i) info badge icon in header right
        painter.setFont(QFont("JetBrains Mono, monospace", 9, QFont.Bold))
        painter.setPen(QColor("#a1a1aa"))
        painter.drawText(self.width - 20, 16, "(i)")

        # Node Title
        painter.setFont(QFont("-apple-system, sans-serif", 10, QFont.Bold))
        painter.setPen(QColor("#09090b"))
        # Truncate title if needed
        short_title = self.title if len(self.title) < 22 else self.title[:19] + "..."
        painter.drawText(12, 48, short_title)

        # Status Dot Pill
        dot_color = QColor("#10b981") if self.status == "SUCCESS" else (QColor("#ef4444") if self.status == "ERROR" else QColor("#f59e0b"))
        painter.setBrush(QBrush(dot_color))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(12, 60, 6, 6)

        painter.setFont(QFont("-apple-system, sans-serif", 8, QFont.Medium))
        painter.setPen(QColor("#71717a"))
        painter.drawText(24, 67, self.status)

    def itemChange(self, change: QGraphicsItem.GraphicsItemChange, value: Any) -> Any:
        if change == QGraphicsItem.ItemPositionHasChanged:
            # Notify scene to redraw connected cables
            if self.scene():
                self.scene().update()
        return super().itemChange(change, value)
