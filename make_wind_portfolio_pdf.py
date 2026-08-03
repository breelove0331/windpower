from __future__ import annotations

import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(".")
SOURCE_MD = ROOT / "portfolio_wind_power_v28_group3_report.md"
OUT_DIR = ROOT / "output" / "pdf"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PDF = OUT_DIR / "wind_power_v28_group3_portfolio_report_v3_final.pdf"

FONT_REGULAR = r"C:\Windows\Fonts\malgun.ttf"
FONT_BOLD = r"C:\Windows\Fonts\malgunbd.ttf"


def register_fonts() -> tuple[str, str]:
    regular = "MalgunGothic"
    bold = "MalgunGothicBold"
    pdfmetrics.registerFont(TTFont(regular, FONT_REGULAR))
    pdfmetrics.registerFont(TTFont(bold, FONT_BOLD))
    return regular, bold


def make_styles(font_name: str, bold_name: str):
    base = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle(
            "title",
            parent=base["Title"],
            fontName=bold_name,
            fontSize=19,
            leading=25,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#1F2937"),
            spaceAfter=10 * mm,
            wordWrap="CJK",
        ),
        "h1": ParagraphStyle(
            "h1",
            parent=base["Heading1"],
            fontName=bold_name,
            fontSize=14,
            leading=20,
            textColor=colors.HexColor("#111827"),
            spaceBefore=6 * mm,
            spaceAfter=3 * mm,
            wordWrap="CJK",
        ),
        "h2": ParagraphStyle(
            "h2",
            parent=base["Heading2"],
            fontName=bold_name,
            fontSize=11.5,
            leading=17,
            textColor=colors.HexColor("#1F2937"),
            spaceBefore=4 * mm,
            spaceAfter=2 * mm,
            wordWrap="CJK",
        ),
        "body": ParagraphStyle(
            "body",
            parent=base["BodyText"],
            fontName=font_name,
            fontSize=9.2,
            leading=14,
            alignment=TA_JUSTIFY,
            textColor=colors.HexColor("#111827"),
            spaceAfter=2.4 * mm,
            wordWrap="CJK",
        ),
        "bullet": ParagraphStyle(
            "bullet",
            parent=base["BodyText"],
            fontName=font_name,
            fontSize=9,
            leading=13.5,
            leftIndent=5 * mm,
            firstLineIndent=0,
            wordWrap="CJK",
        ),
        "caption": ParagraphStyle(
            "caption",
            parent=base["BodyText"],
            fontName=font_name,
            fontSize=8,
            leading=11,
            textColor=colors.HexColor("#4B5563"),
            alignment=TA_CENTER,
            spaceBefore=1.5 * mm,
            spaceAfter=3 * mm,
            wordWrap="CJK",
        ),
        "table_cell": ParagraphStyle(
            "table_cell",
            parent=base["BodyText"],
            fontName=font_name,
            fontSize=7.3,
            leading=9.5,
            wordWrap="CJK",
        ),
        "table_header": ParagraphStyle(
            "table_header",
            parent=base["BodyText"],
            fontName=bold_name,
            fontSize=7.2,
            leading=9.5,
            textColor=colors.white,
            alignment=TA_CENTER,
            wordWrap="CJK",
        ),
    }
    return styles


def clean_inline(text: str) -> str:
    text = html.escape(text.strip())
    text = re.sub(r"`([^`]+)`", r"<font name='MalgunGothicBold'>\1</font>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    return text


def parse_table(lines: list[str], start: int):
    table_lines = []
    i = start
    while i < len(lines) and lines[i].strip().startswith("|"):
        table_lines.append(lines[i].strip())
        i += 1
    if len(table_lines) < 2:
        return None, start
    rows = []
    for line in table_lines:
        cells = [c.strip() for c in line.strip("|").split("|")]
        if all(re.fullmatch(r":?-{3,}:?", c or "") for c in cells):
            continue
        rows.append(cells)
    return rows, i


def table_col_widths(n_cols: int, page_width: float):
    if n_cols <= 2:
        return [page_width * 0.42, page_width * 0.58]
    if n_cols == 3:
        return [page_width * 0.42, page_width * 0.29, page_width * 0.29]
    return [page_width / n_cols] * n_cols


def build_table(rows: list[list[str]], styles, page_width: float):
    max_cols = max(len(r) for r in rows)
    normalized = [r + [""] * (max_cols - len(r)) for r in rows]
    data = []
    for ridx, row in enumerate(normalized):
        style = styles["table_header"] if ridx == 0 else styles["table_cell"]
        data.append([Paragraph(clean_inline(c), style) for c in row])
    table = Table(data, colWidths=table_col_widths(max_cols, page_width), repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#334155")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD5E1")),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F8FAFC")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return KeepTogether([Spacer(1, 1.5 * mm), table, Spacer(1, 3 * mm)])


def markdown_to_flowables(text: str, styles, page_width: float):
    lines = text.splitlines()
    story = []
    bullets = []
    i = 0

    def flush_bullets():
        nonlocal bullets
        if bullets:
            items = [
                ListItem(Paragraph(clean_inline(item), styles["bullet"]), bulletColor=colors.HexColor("#475569"))
                for item in bullets
            ]
            story.append(ListFlowable(items, bulletType="bullet", start="circle", leftIndent=6 * mm))
            story.append(Spacer(1, 1.5 * mm))
            bullets = []

    while i < len(lines):
        raw = lines[i]
        line = raw.strip()
        if not line:
            flush_bullets()
            i += 1
            continue

        if line.startswith("|"):
            flush_bullets()
            rows, next_i = parse_table(lines, i)
            if rows:
                story.append(build_table(rows, styles, page_width))
                i = next_i
                continue

        if line.startswith("# "):
            flush_bullets()
            title = line[2:].strip()
            story.append(Paragraph(clean_inline(title), styles["title"]))
            story.append(Paragraph("DACON 풍력발전량 예측 경진대회 분석 포트폴리오", styles["caption"]))
        elif line.startswith("## "):
            flush_bullets()
            story.append(Paragraph(clean_inline(line[3:]), styles["h1"]))
        elif line.startswith("### "):
            flush_bullets()
            story.append(Paragraph(clean_inline(line[4:]), styles["h2"]))
        elif line.startswith("- "):
            bullets.append(line[2:].strip())
        else:
            flush_bullets()
            story.append(Paragraph(clean_inline(line), styles["body"]))
        i += 1
    flush_bullets()
    return story


def footer(canvas, doc):
    canvas.saveState()
    width, _ = A4
    canvas.setFont("MalgunGothic", 7.5)
    canvas.setFillColor(colors.HexColor("#64748B"))
    canvas.drawString(18 * mm, 10 * mm, "Wind Power Forecasting Portfolio Report")
    canvas.drawRightString(width - 18 * mm, 10 * mm, f"{doc.page}")
    canvas.restoreState()


def main():
    if not SOURCE_MD.exists():
        raise FileNotFoundError(f"Missing source markdown: {SOURCE_MD}")
    font_name, bold_name = register_fonts()
    styles = make_styles(font_name, bold_name)

    page_width, page_height = A4
    margin_x = 17 * mm
    margin_top = 16 * mm
    margin_bottom = 16 * mm
    frame = Frame(
        margin_x,
        margin_bottom,
        page_width - 2 * margin_x,
        page_height - margin_top - margin_bottom,
        id="normal",
    )

    doc = BaseDocTemplate(
        str(OUT_PDF),
        pagesize=A4,
        leftMargin=margin_x,
        rightMargin=margin_x,
        topMargin=margin_top,
        bottomMargin=margin_bottom,
        title="Wind Power Forecasting Portfolio Report",
        author="Codex",
    )
    doc.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=footer)])

    story = markdown_to_flowables(SOURCE_MD.read_text(encoding="utf-8"), styles, page_width - 2 * margin_x)
    doc.build(story)
    print(OUT_PDF)


if __name__ == "__main__":
    main()
