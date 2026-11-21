"""
Decision Graph Application Entry Point.
Launches the standalone PySide6 desktop window with high-DPI scaling.
"""

import sys
import os
import platform

# Fast bypass for Windows WMI COM lockup
def _fast_wmi_bypass(*args, **kwargs):
    raise OSError("WMI bypassed for instant execution")
platform._wmi_query = _fast_wmi_bypass

from pathlib import Path

# Ensure project root is in python path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from src.app import MainWindow


def main():
    # Enable High-DPI display scaling
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"

    app = QApplication(sys.argv)
    app.setApplicationName("Decision Graph")
    app.setOrganizationName("Enterprise Analytics")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
