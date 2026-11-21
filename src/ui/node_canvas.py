"""
Interactive Visual Node Canvas.
Provides infinite pan, zoom, draggable nodes, and cubic bezier cable connections.
Emits signals on node selection, connection creation, and deletion.
"""

from typing import Dict, List, Optional, Tuple
from PySide6.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsPathItem, QWidget
)
from PySide6.QtCore import Qt, QPointF, Signal, QRectF
from PySide6.QtGui import (
    QPainter, QPen, QColor, QPainterPath, QBrush
)

from src.ui.node_item import NodeCardItem, PortItem


class ConnectionCableItem(QGraphicsPathItem):
    """Cubic Bezier Connection Cable between two nodes."""

    def __init__(self, source_port: PortItem, target_port: PortItem):
        super().__init__()
        self.source_port = source_port
        self.target_port = target_port
        self.setZValue(-1)  # Place cable behind nodes

        pen = QPen(QColor("#71717a"), 2.0, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        self.setPen(pen)
        self.update_path()

    def update_path(self):
        p1 = self.source_port.get_scene_pos()
        p2 = self.target_port.get_scene_pos()

        path = QPainterPath(p1)
        # Bezier control points for natural horizontal flow
        dx = max(40.0, abs(p2.x() - p1.x()) * 0.5)
        c1 = QPointF(p1.x() + dx, p1.y())
        c2 = QPointF(p2.x() - dx, p2.y())
        path.cubicTo(c1, c2, p2)
        self.setPath(path)


class NodeCanvasScene(QGraphicsScene):
    """Custom GraphicsScene handling high-performance dot-grid rendering."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setSceneRect(-3000, -3000, 6000, 6000)
        self.grid_size = 24

        # Pre-render 24x24 dot pattern into cached QPixmap for 60 FPS hardware accelerated tiling
        from PySide6.QtGui import QPixmap
        tile = QPixmap(self.grid_size, self.grid_size)
        tile.fill(QColor("#f8fafc"))
        tile_painter = QPainter(tile)
        tile_painter.setRenderHint(QPainter.Antialiasing)
        tile_painter.setPen(Qt.NoPen)
        tile_painter.setBrush(QColor("#cbd5e1"))
        tile_painter.drawEllipse(11, 11, 2, 2)
        tile_painter.end()
        self._bg_brush = QBrush(tile)

    def drawBackground(self, painter: QPainter, rect: QRectF):
        painter.fillRect(rect, self._bg_brush)


class NodeCanvasView(QGraphicsView):
    """High-performance canvas view supporting zoom, pan, and reactive node wiring."""

    node_selected = Signal(str)            # node_id
    connection_requested = Signal(str, str) # (source_node_id, target_node_id)
    node_deleted = Signal(str)             # node_id

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.scene_obj = NodeCanvasScene(self)
        self.setScene(self.scene_obj)
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setViewportUpdateMode(QGraphicsView.SmartViewportUpdate)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)

        self.nodes_map: Dict[str, NodeCardItem] = {}
        self.cables: List[ConnectionCableItem] = []

        # Interactive Cable Dragging State
        self.drag_cable_source: Optional[PortItem] = None
        self.drag_cable_path: Optional[QGraphicsPathItem] = None

        # Pan state
        self._is_panning = False
        self._pan_start = QPointF()

    def add_node_item(self, node_id: str, node_type: str, title: str, x: float = 100.0, y: float = 100.0) -> NodeCardItem:
        item = NodeCardItem(node_id, node_type, title, x, y)
        self.scene_obj.addItem(item)
        self.nodes_map[node_id] = item
        return item

    def update_node_status(self, node_id: str, status: str):
        if node_id in self.nodes_map:
            self.nodes_map[node_id].status = status
            self.nodes_map[node_id].update()

    def add_connection_item(self, source_id: str, target_id: str):
        src_item = self.nodes_map.get(source_id)
        tgt_item = self.nodes_map.get(target_id)
        if src_item and tgt_item:
            cable = ConnectionCableItem(src_item.output_port, tgt_item.input_port)
            self.scene_obj.addItem(cable)
            self.cables.append(cable)

    def mousePressEvent(self, event):
        pos = event.pos()
        scene_pos = self.mapToScene(pos)
        item = self.scene_obj.itemAt(scene_pos, self.transform())

        # Middle Click or Alt+Click for panning
        if event.button() == Qt.MiddleButton or (event.button() == Qt.LeftButton and event.modifiers() & Qt.AltModifier):
            self._is_panning = True
            self._pan_start = event.position()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return

        # Port Click: Begin cable dragging
        if isinstance(item, PortItem) and not item.is_input:
            self.drag_cable_source = item
            self.drag_cable_path = QGraphicsPathItem()
            self.drag_cable_path.setPen(QPen(QColor("#10b981"), 2.0, Qt.DashLine))
            self.scene_obj.addItem(self.drag_cable_path)
            event.accept()
            return

        # Select Node
        if isinstance(item, NodeCardItem):
            self.node_selected.emit(item.node_id)
        elif item and item.parentItem() and isinstance(item.parentItem(), NodeCardItem):
            self.node_selected.emit(item.parentItem().node_id)

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._is_panning:
            delta = event.position() - self._pan_start
            self._pan_start = event.position()
            self.horizontalScrollBar().setValue(int(self.horizontalScrollBar().value() - delta.x()))
            self.verticalScrollBar().setValue(int(self.verticalScrollBar().value() - delta.y()))
            event.accept()
            return

        # Update dragged cable
        if self.drag_cable_source and self.drag_cable_path:
            p1 = self.drag_cable_source.get_scene_pos()
            p2 = self.mapToScene(event.pos())
            path = QPainterPath(p1)
            dx = max(30.0, abs(p2.x() - p1.x()) * 0.5)
            path.cubicTo(QPointF(p1.x() + dx, p1.y()), QPointF(p2.x() - dx, p2.y()), p2)
            self.drag_cable_path.setPath(path)
            event.accept()
            return

        # Redraw existing cables on node movement
        for c in self.cables:
            c.update_path()

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._is_panning:
            self._is_panning = False
            self.setCursor(Qt.ArrowCursor)
            event.accept()
            return

        # Connect cable if dropped on input port
        if self.drag_cable_source and self.drag_cable_path:
            scene_pos = self.mapToScene(event.pos())
            target_item = self.scene_obj.itemAt(scene_pos, self.transform())
            
            if isinstance(target_item, PortItem) and target_item.is_input:
                src_node = self.drag_cable_source.parent_node.node_id
                tgt_node = target_item.parent_node.node_id
                self.connection_requested.emit(src_node, tgt_node)

            # Clean up drag path
            self.scene_obj.removeItem(self.drag_cable_path)
            self.drag_cable_path = None
            self.drag_cable_source = None
            event.accept()
            return

        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        """Smooth Canvas Zooming."""
        factor = 1.15 if event.angleDelta().y() > 0 else 0.85
        self.scale(factor, factor)
        event.accept()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Delete or event.key() == Qt.Key_Backspace:
            for item in self.scene_obj.selectedItems():
                if isinstance(item, NodeCardItem):
                    self.node_deleted.emit(item.node_id)
                    self.scene_obj.removeItem(item)
                    del self.nodes_map[item.node_id]
            # Redraw cables
            self.cables = [c for c in self.cables if c.source_port.parent_node.node_id in self.nodes_map and c.target_port.parent_node.node_id in self.nodes_map]
            self.scene_obj.update()
            event.accept()
            return
        super().keyPressEvent(event)
