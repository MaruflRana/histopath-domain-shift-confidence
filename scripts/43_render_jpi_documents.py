"""Fallback DOCX-to-PDF/page renderer for Milestone 9D visual QA.

This utility is used only because Word/LibreOffice is unavailable. It reads the generated
DOCX files, renders their visible paragraph/table content with ReportLab, and rasterizes the
review PDFs with PDFium. It imports no dataset, model, training, or evaluation module.
"""

import os
from pathlib import Path
import re
import shutil

import pypdfium2 as pdfium
from docx import Document
from docx.document import Document as DocumentClass
from docx.table import Table as DocxTable
from docx.text.paragraph import Paragraph as DocxParagraph
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "submission" / "jpi"
RENDER = ROOT / "tmp" / "jpi_render"


def escape(text):
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def register_fonts():
    windows_dir = os.environ.get("WINDIR")
    if not windows_dir:
        return
    fonts_dir = Path(windows_dir) / "Fonts"
    candidates = [
        ("Calibri", fonts_dir / "calibri.ttf"),
        ("Calibri-Bold", fonts_dir / "calibrib.ttf"),
        ("Calibri-Italic", fonts_dir / "calibrii.ttf"),
    ]
    for name, path in candidates:
        if path.exists():
            pdfmetrics.registerFont(TTFont(name, str(path)))


def iter_blocks(document):
    parent = document.element.body
    for child in parent.iterchildren():
        if isinstance(child, CT_P):
            yield DocxParagraph(child, document)
        elif isinstance(child, CT_Tbl):
            yield DocxTable(child, document)


def paragraph_markup(paragraph):
    parts = []
    for run in paragraph.runs:
        text = escape(run.text)
        if not text:
            continue
        if run.bold:
            text = f"<b>{text}</b>"
        if run.italic:
            text = f"<i>{text}</i>"
        if run.font.superscript:
            text = f"<super>{text}</super>"
        parts.append(text)
    return "".join(parts) or escape(paragraph.text)


def styles():
    base = getSampleStyleSheet()
    font = "Calibri" if "Calibri" in pdfmetrics.getRegisteredFontNames() else "Helvetica"
    bold = "Calibri-Bold" if "Calibri-Bold" in pdfmetrics.getRegisteredFontNames() else "Helvetica-Bold"
    italic = "Calibri-Italic" if "Calibri-Italic" in pdfmetrics.getRegisteredFontNames() else "Helvetica-Oblique"
    return {
        "normal": ParagraphStyle("JPI_Normal", parent=base["BodyText"], fontName=font, fontSize=10.5, leading=13.5, alignment=TA_JUSTIFY, spaceAfter=7),
        "title": ParagraphStyle("JPI_Title", parent=base["Title"], fontName=bold, fontSize=17, leading=20, textColor=colors.HexColor("#1F4D78"), alignment=TA_CENTER, spaceAfter=12),
        "h1": ParagraphStyle("JPI_H1", parent=base["Heading1"], fontName=bold, fontSize=14, leading=17, textColor=colors.HexColor("#2E74B5"), spaceBefore=12, spaceAfter=7),
        "h2": ParagraphStyle("JPI_H2", parent=base["Heading2"], fontName=bold, fontSize=12, leading=14, textColor=colors.HexColor("#2E74B5"), spaceBefore=9, spaceAfter=5),
        "h3": ParagraphStyle("JPI_H3", parent=base["Heading3"], fontName=bold, fontSize=11, leading=13, textColor=colors.HexColor("#1F4D78"), spaceBefore=7, spaceAfter=4),
        "bullet": ParagraphStyle("JPI_Bullet", parent=base["BodyText"], fontName=font, fontSize=10.5, leading=13, leftIndent=16, firstLineIndent=-9, bulletIndent=6, spaceAfter=4),
        "table": ParagraphStyle("JPI_Table", parent=base["BodyText"], fontName=font, fontSize=6.5, leading=8, alignment=TA_LEFT),
        "table_head": ParagraphStyle("JPI_Table_Head", parent=base["BodyText"], fontName=bold, fontSize=6.5, leading=8, alignment=TA_LEFT),
        "italic": ParagraphStyle("JPI_Italic", parent=base["BodyText"], fontName=italic, fontSize=9, leading=11, spaceAfter=6),
    }


def render_table(table, style_map, available_width):
    raw = [[cell.text.strip() for cell in row.cells] for row in table.rows]
    if not raw:
        return Spacer(1, 1)
    columns = max(len(row) for row in raw)
    widths = []
    for col in range(columns):
        maximum = max(len(row[col]) if col < len(row) else 0 for row in raw)
        widths.append(max(5, min(maximum, 35)))
    total = sum(widths)
    col_widths = [available_width * value / total for value in widths]
    content = []
    for row_index, row in enumerate(raw):
        row_cells = []
        for col in range(columns):
            value = row[col] if col < len(row) else ""
            row_cells.append(Paragraph(escape(value), style_map["table_head"] if row_index == 0 else style_map["table"]))
        content.append(row_cells)
    rendered = Table(content, colWidths=col_widths, repeatRows=1, hAlign="CENTER")
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 2),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2),
        ("LINEABOVE", (0, 0), (-1, 0), 0.6, colors.HexColor("#666666")),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.HexColor("#666666")),
        ("LINEBELOW", (0, -1), (-1, -1), 0.6, colors.HexColor("#666666")),
    ]
    rendered.setStyle(TableStyle(commands))
    return rendered


def add_page_number(canvas, document):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawCentredString(letter[0] / 2, 0.45 * inch, str(document.page))
    canvas.restoreState()


def render_docx(path, output_dir):
    document = Document(path)
    style_map = styles()
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{path.stem}.pdf"
    anonymous = path.name in {
        "JPI_Anonymized_Manuscript.docx", "JPI_Figure_Captions.docx", "JPI_Tables.docx",
        "JPI_Supplementary_Material.docx", "JPI_CLAIM_Checklist.docx",
    }
    pdf = SimpleDocTemplate(
        str(pdf_path), pagesize=letter, rightMargin=0.72 * inch, leftMargin=0.72 * inch,
        topMargin=0.72 * inch, bottomMargin=0.65 * inch,
        title=document.core_properties.title or path.stem,
        author="Anonymous" if anonymous else "Jishan Islam Maruf; Ishtiak Al Mamoon",
        subject=document.core_properties.subject or "Journal submission document",
    )
    story = []
    for block in iter_blocks(document):
        if isinstance(block, DocxParagraph):
            text = paragraph_markup(block)
            has_page_break = 'w:type="page"' in block._p.xml
            if has_page_break:
                story.append(PageBreak())
            if not text.strip():
                if has_page_break:
                    continue
                story.append(Spacer(1, 3))
                continue
            style_name = block.style.name if block.style is not None else "Normal"
            if style_name == "Title":
                chosen = style_map["title"]
            elif style_name == "Heading 1":
                chosen = style_map["h1"]
            elif style_name == "Heading 2":
                chosen = style_map["h2"]
            elif style_name == "Heading 3":
                chosen = style_map["h3"]
            elif style_name.startswith("List Bullet"):
                story.append(Paragraph(text, style_map["bullet"], bulletText="•"))
                continue
            elif style_name.startswith("List Number"):
                chosen = style_map["bullet"]
            else:
                chosen = style_map["normal"]
            if block.paragraph_format.page_break_before:
                story.append(PageBreak())
            story.append(Paragraph(text, chosen))
        else:
            story.append(render_table(block, style_map, pdf.width))
            story.append(Spacer(1, 7))
    pdf.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
    pdf_doc = pdfium.PdfDocument(str(pdf_path))
    for index in range(len(pdf_doc)):
        page = pdf_doc[index]
        bitmap = page.render(scale=2.0, rotation=0)
        image = bitmap.to_pil()
        image.save(output_dir / f"page_{index + 1:03d}.png", dpi=(144, 144))
    return pdf_path, len(pdf_doc)


def main(only=None):
    if RENDER.exists():
        if RENDER.resolve().parent != (ROOT / "tmp").resolve():
            raise RuntimeError("Unexpected render directory")
        shutil.rmtree(RENDER)
    RENDER.mkdir(parents=True)
    register_fonts()
    report = []
    paths = sorted(SOURCE.glob("*.docx"))
    if only:
        paths = [path for path in paths if path.name == only]
        if not paths:
            raise FileNotFoundError(f"Requested DOCX not found: {only}")
    for path in paths:
        output = RENDER / path.stem
        pdf_path, pages = render_docx(path, output)
        report.append(f"{path.name}\t{pages}\t{pdf_path.relative_to(ROOT).as_posix()}")
        if path.name == "JPI_Anonymized_Manuscript.docx":
            shutil.copy2(pdf_path, SOURCE / "JPI_Anonymized_Manuscript.pdf")
    (RENDER / "render_manifest.txt").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Render JPI DOCX files with the document-only fallback renderer.")
    parser.add_argument("--only", help="Render only the named DOCX file.")
    args = parser.parse_args()
    main(only=args.only)
