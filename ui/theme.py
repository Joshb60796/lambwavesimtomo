"""Anodized-bench palette and the two faces used on the panel."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import QFont, QFontDatabase

ROOT = Path(__file__).resolve().parents[1]
FONT_DIR = ROOT / "assets" / "fonts"

INK = "#14160f"
PANEL = "#1c1f17"
PANEL_2 = "#24281e"
LINE = "#3c4133"
TEXT = "#e7e1d4"
MUTED = "#9a9486"
AMBER = "#e2a122"
PHOSPHOR = "#c6dc78"
TEAL = "#7fafa3"
OXIDE = "#d4653c"
ALUMINUM = "#c8c2b4"

UI_FAMILY = "Barlow Condensed"
MONO_FAMILY = "Share Tech Mono"


def load_fonts() -> None:
    for path in sorted(FONT_DIR.glob("*.ttf")):
        QFontDatabase.addApplicationFont(str(path))


_WEIGHTS = {
    100: QFont.Weight.Thin,
    200: QFont.Weight.ExtraLight,
    300: QFont.Weight.Light,
    400: QFont.Weight.Normal,
    500: QFont.Weight.Medium,
    600: QFont.Weight.DemiBold,
    700: QFont.Weight.Bold,
    800: QFont.Weight.ExtraBold,
    900: QFont.Weight.Black,
}


def ui_font(size: int, weight: QFont.Weight | int = QFont.Weight.Medium) -> QFont:
    font = QFont(UI_FAMILY, size)
    if not isinstance(weight, QFont.Weight):
        weight = _WEIGHTS.get(int(weight), QFont.Weight.Medium)
    font.setWeight(weight)
    font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
    return font


def mono_font(size: int) -> QFont:
    font = QFont(MONO_FAMILY, size)
    font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
    return font


def stylesheet() -> str:
    return f"""
    QMainWindow, QWidget {{
        background: {INK};
        color: {TEXT};
        font-family: "{UI_FAMILY}";
        font-size: 15px;
    }}
    QWidget#panel {{
        background: {PANEL};
        border: 1px solid {LINE};
    }}
    QLabel#title {{
        font-size: 30px;
        font-weight: 600;
        color: {TEXT};
        background: transparent;
    }}
    QLabel#subtitle {{
        color: {MUTED};
        font-size: 15px;
        background: transparent;
    }}
    QLabel#kicker {{
        color: {AMBER};
        font-size: 13px;
        font-weight: 600;
        background: transparent;
    }}
    QLabel#muted {{
        color: {MUTED};
        font-size: 13px;
        background: transparent;
    }}
    QLabel#chip {{
        background: {PANEL_2};
        border: 1px solid {LINE};
        color: {ALUMINUM};
        font-family: "{MONO_FAMILY}";
        font-size: 11px;
        padding: 3px 7px;
    }}
    QLabel#readout {{
        font-family: "{MONO_FAMILY}";
        font-size: 12px;
        color: {TEXT};
        background: transparent;
    }}
    QPushButton {{
        background: {PANEL_2};
        color: {TEXT};
        border: 1px solid #515743;
        padding: 7px 12px;
        font-weight: 600;
        font-size: 14px;
    }}
    QPushButton:hover {{
        border-color: {AMBER};
        color: #f6e2bc;
    }}
    QPushButton:pressed {{
        background: #3a3016;
    }}
    QPushButton:disabled {{
        color: #6d6a60;
        border-color: #33362c;
    }}
    QPushButton#primary {{
        background: #3a2d12;
        border: 1px solid {AMBER};
        color: #f3ddb0;
    }}
    QPushButton#preset {{
        padding: 5px 8px;
        font-size: 13px;
    }}
    QSlider::groove:horizontal {{
        height: 3px;
        background: #3a3f32;
        border: none;
    }}
    QSlider::handle:horizontal {{
        width: 11px;
        height: 14px;
        margin: -6px 0;
        background: {AMBER};
        border: none;
    }}
    QSlider::sub-page:horizontal {{
        background: #8a6418;
    }}
    QTableWidget {{
        background: #12140f;
        alternate-background-color: #181b14;
        color: {TEXT};
        gridline-color: #2a2e24;
        selection-background-color: #3d3318;
        selection-color: #f6e7c8;
        font-family: "{MONO_FAMILY}";
        font-size: 11px;
        border: none;
    }}
    QHeaderView::section {{
        background: {PANEL};
        color: {MUTED};
        border: none;
        border-bottom: 1px solid {LINE};
        padding: 4px 6px;
        font-family: "{UI_FAMILY}";
        font-size: 12px;
        font-weight: 600;
    }}
    QStatusBar {{
        background: #10120c;
        color: {MUTED};
        font-family: "{MONO_FAMILY}";
        font-size: 11px;
    }}
    QSplitter::handle {{
        background: {INK};
    }}
    QScrollBar:vertical {{
        background: {PANEL};
        width: 10px;
        margin: 0;
    }}
    QScrollBar::handle:vertical {{
        background: #3e4436;
        min-height: 24px;
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0;
    }}
    """
