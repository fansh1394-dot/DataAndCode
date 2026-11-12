# -*- coding: utf-8 -*-
"""把《双光谱反无人机检测_小论文选题方案_v3.md》转成 docx。

排版规范沿用《六个SCI二区小论文选题详细研究与实验方案》(v2) 的 build_v2_part1.py：
  标题 微软雅黑 16/13/11.5pt；正文 宋体 + Times New Roman 11pt，行距 1.35，首行缩进 22pt；
  表格表头深蓝底白字、隔行浅蓝底、表头跨页重复；
  代码块用等宽字体 + 浅灰底纹单元格。
Markdown 支持：ATX 标题、管道表格、围栏代码块、有序/无序列表、引用块、粗体与行内代码。
"""
import re
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

# 复用 v2 排版工具
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_v2_part1 import (  # noqa: E402
    BLACK, LIGHT_GRAY, MID_GRAY, NAVY, PALE_BLUE, SOFT_YELLOW, WHITE,
    add_body, add_heading, set_cell_margins, set_cell_shading,
    set_run_font, set_table_borders, repeat_header, cant_split,
)

BASE = Path(r"C:\Users\不如喝茶去\Desktop\双光谱毕设\子方向选题")
SRC = BASE / "双光谱反无人机检测_小论文选题方案_v3.md"
OUT = BASE / "双光谱反无人机检测_小论文选题方案_v3.docx"

CODE_FILL = "F4F6F8"
QUOTE_FILL = "FDF6E3"


def add_table_md(doc, headers, rows, font_size=9.0):
    """Markdown 表格 → Word 表格（自动列宽，含表头样式）。"""
    ncol = len(headers)
    table = doc.add_table(rows=1, cols=ncol)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True
    set_table_borders(table)
    hdr = table.rows[0]
    repeat_header(hdr)
    for i, h in enumerate(headers):
        cell = hdr.cells[i]
        set_cell_shading(cell, NAVY)
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        set_cell_margins(cell, top=70, start=85, bottom=70, end=85)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.line_spacing = 1.08
        r = p.add_run(str(h))
        set_run_font(r, east="微软雅黑", west="Arial", size=font_size, bold=True, color=WHITE)
    for ridx, row in enumerate(rows):
        cells = table.add_row().cells
        cant_split(table.rows[-1])
        for i in range(ncol):
            val = row[i] if i < len(row) else ""
            cell = cells[i]
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            set_cell_margins(cell, top=70, start=85, bottom=70, end=85)
            if ridx % 2 == 1:
                set_cell_shading(cell, PALE_BLUE)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if (i == 0 or len(val) <= 12) else WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.12
            add_rich_runs(p, val, size=font_size)
    sp = doc.add_paragraph()
    sp.paragraph_format.space_after = Pt(2)
    return table


def add_code_block(doc, lines):
    """围栏代码块 → 单格浅灰表，等宽字体，保留缩进。"""
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    set_table_borders(table, color="C9D6E4", size="6")
    cell = table.rows[0].cells[0]
    cell.width = Inches(6.9)
    set_cell_shading(cell, CODE_FILL)
    set_cell_margins(cell, top=90, start=130, bottom=90, end=130)
    first = True
    for ln in lines:
        p = cell.paragraphs[0] if first else cell.add_paragraph()
        first = False
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.line_spacing = 1.12
        r = p.add_run(ln if ln else "")
        set_run_font(r, east="宋体", west="Consolas", size=8.6, color="1A1A1A")
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def add_callout(doc, text, fill=QUOTE_FILL):
    """引用块 → 带底纹强调段。"""
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    set_table_borders(table, color="C9D6E4", size="6")
    cell = table.rows[0].cells[0]
    cell.width = Inches(6.9)
    set_cell_shading(cell, fill)
    set_cell_margins(cell, top=100, start=140, bottom=100, end=140)
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.line_spacing = 1.3
    add_rich_runs(p, text)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


INLINE_RE = re.compile(r"(\*\*[^*]+\*\*|`[^`]+`)")


def add_rich_runs(p, text, size=11):
    """解析 **粗体** 与 `行内代码`。"""
    for seg in INLINE_RE.split(text):
        if not seg:
            continue
        if seg.startswith("**") and seg.endswith("**") and len(seg) > 4:
            r = p.add_run(seg[2:-2])
            set_run_font(r, size=size, bold=True)
        elif seg.startswith("`") and seg.endswith("`") and len(seg) > 2:
            r = p.add_run(seg[1:-1])
            set_run_font(r, east="宋体", west="Consolas", size=size - 0.6, color="B03A2E")
        else:
            r = p.add_run(seg)
            set_run_font(r, size=size)


def add_bullet_md(doc, text, level=0):
    style = "List Bullet" if level == 0 else "List Bullet 2"
    p = doc.add_paragraph(style=style)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.line_spacing = 1.25
    add_rich_runs(p, text)


def add_numbered_md(doc, number, text, level=0):
    p = doc.add_paragraph(style="Normal")
    p.paragraph_format.left_indent = Pt(22 + 16 * level)
    p.paragraph_format.first_line_indent = Pt(-18)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.line_spacing = 1.25
    add_rich_runs(p, f"{number}. {text}")


def setup_doc():
    doc = Document()
    sec = doc.sections[0]
    sec.top_margin = Inches(0.85)
    sec.bottom_margin = Inches(0.85)
    sec.left_margin = Inches(0.8)
    sec.right_margin = Inches(0.8)
    style = doc.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(11)
    style._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    return doc


def add_page_number_footer(doc):
    p = doc.sections[0].footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    f1 = OxmlElement("w:fldChar"); f1.set(qn("w:fldCharType"), "begin")
    it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve"); it.text = " PAGE "
    f2 = OxmlElement("w:fldChar"); f2.set(qn("w:fldCharType"), "end")
    run._r.extend([f1, it, f2])
    set_run_font(run, size=9, color=MID_GRAY)


def build():
    md = SRC.read_text(encoding="utf-8").splitlines()
    doc = setup_doc()

    # 封面
    for _ in range(4):
        doc.add_paragraph()
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("双光谱反无人机检测")
    set_run_font(r, east="微软雅黑", west="Arial", size=26, bold=True, color=NAVY)
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("小论文选题方案（v3）")
    set_run_font(r, east="微软雅黑", west="Arial", size=22, bold=True, color=NAVY)
    for _ in range(2):
        doc.add_paragraph()
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("四个可发表方向 · 完整选题论证 · 实验矩阵 · 技术路线")
    set_run_font(r, east="微软雅黑", west="Arial", size=12, color=MID_GRAY)
    for _ in range(6):
        doc.add_paragraph()
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("2026 年 10 月")
    set_run_font(r, size=11)
    doc.add_page_break()

    i = 0
    n = len(md)
    in_code = False
    code_buf = []
    while i < n:
        line = md[i]
        stripped = line.strip()

        # 围栏代码块
        if stripped.startswith("```"):
            if in_code:
                add_code_block(doc, code_buf)
                code_buf, in_code = [], False
            else:
                in_code = True
            i += 1
            continue
        if in_code:
            code_buf.append(line.rstrip())
            i += 1
            continue

        # 分隔线
        if re.fullmatch(r"-{3,}", stripped):
            i += 1
            continue

        # 标题
        m = re.match(r"^(#{1,4})\s+(.*)$", stripped)
        if m:
            level = len(m.group(1))
            text = m.group(2).strip()
            if level == 1:
                doc.add_page_break()
                add_heading(doc, text, 1)
            else:
                add_heading(doc, text, min(level - 1, 3))
            i += 1
            continue

        # 表格
        if stripped.startswith("|") and i + 1 < n and re.match(r"^\|[\s:\-|]+\|$", md[i + 1].strip()):
            headers = [c.strip() for c in stripped.strip("|").split("|")]
            i += 2
            rows = []
            while i < n and md[i].strip().startswith("|"):
                cells = [c.strip() for c in md[i].strip().strip("|").split("|")]
                cells = [c.replace("<br>", " / ") for c in cells]
                rows.append(cells)
                i += 1
            add_table_md(doc, headers, rows)
            continue

        # 引用块
        if stripped.startswith(">"):
            buf = []
            while i < n and md[i].strip().startswith(">"):
                buf.append(md[i].strip().lstrip(">").strip())
                i += 1
            add_callout(doc, " ".join(x for x in buf if x))
            continue

        # 无序列表
        m = re.match(r"^(\s*)-\s+(.*)$", line)
        if m:
            level = 1 if len(m.group(1)) >= 2 else 0
            add_bullet_md(doc, m.group(2).strip(), level)
            i += 1
            continue

        # 有序列表
        m = re.match(r"^(\s*)(\d+)\.\s+(.*)$", line)
        if m:
            level = 1 if len(m.group(1)) >= 2 else 0
            add_numbered_md(doc, m.group(2), m.group(3).strip(), level)
            i += 1
            continue

        # 空行
        if not stripped:
            i += 1
            continue

        # 普通段落：合并连续非空行
        buf = [stripped]
        i += 1
        while i < n:
            nxt = md[i].strip()
            if (not nxt or nxt.startswith("#") or nxt.startswith("|") or nxt.startswith(">")
                    or nxt.startswith("```") or re.match(r"^\s*[-*]\s+", md[i])
                    or re.match(r"^\s*\d+\.\s+", md[i]) or re.fullmatch(r"-{3,}", nxt)):
                break
            buf.append(nxt)
            i += 1
        add_body(doc, " ".join(buf), first_indent=True)

    if in_code and code_buf:
        add_code_block(doc, code_buf)

    add_page_number_footer(doc)
    doc.save(str(OUT))
    print("[OK]", OUT)
    print("段落数:", len(doc.paragraphs), "表格数:", len(doc.tables))


if __name__ == "__main__":
    build()
