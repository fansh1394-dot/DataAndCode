from pathlib import Path
from datetime import date

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


OUT_DIR = Path(r"C:\Users\不如喝茶去\Desktop\双光谱毕设\子方向选题\子方向选题报告")
OUT_FILE = OUT_DIR / "六个SCI二区小论文选题详细研究与实验方案.docx"

NAVY = "1F4E78"
PALE_BLUE = "EAF2F8"
LIGHT_GRAY = "D9D9D9"
MID_GRAY = "666666"
BLACK = "000000"
WHITE = "FFFFFF"


def set_cell_shading(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_margins(cell, top=90, start=110, bottom=90, end=110):
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for m, v in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{m}"))
        if node is None:
            node = OxmlElement(f"w:{m}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(v))
        node.set(qn("w:type"), "dxa")


def set_table_borders(table, color=LIGHT_GRAY, size="6"):
    tbl_pr = table._tbl.tblPr
    borders = tbl_pr.find(qn("w:tblBorders"))
    if borders is None:
        borders = OxmlElement("w:tblBorders")
        tbl_pr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        tag = borders.find(qn(f"w:{edge}"))
        if tag is None:
            tag = OxmlElement(f"w:{edge}")
            borders.append(tag)
        tag.set(qn("w:val"), "single")
        tag.set(qn("w:sz"), size)
        tag.set(qn("w:space"), "0")
        tag.set(qn("w:color"), color)


def repeat_header(row):
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)


def cant_split(row):
    tr_pr = row._tr.get_or_add_trPr()
    node = OxmlElement("w:cantSplit")
    tr_pr.append(node)


def set_run_font(run, east="宋体", west="Times New Roman", size=11, bold=None, color=BLACK):
    run.font.name = west
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), east)
    run._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), west)
    run._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), west)
    run.font.size = Pt(size)
    run.font.color.rgb = RGBColor.from_string(color)
    if bold is not None:
        run.bold = bold


def add_page_number(paragraph):
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run()
    fld_char1 = OxmlElement("w:fldChar")
    fld_char1.set(qn("w:fldCharType"), "begin")
    instr_text = OxmlElement("w:instrText")
    instr_text.set(qn("xml:space"), "preserve")
    instr_text.text = " PAGE "
    fld_char2 = OxmlElement("w:fldChar")
    fld_char2.set(qn("w:fldCharType"), "end")
    run._r.extend([fld_char1, instr_text, fld_char2])
    set_run_font(run, size=9, color=MID_GRAY)


def add_body(doc, text, bold_lead=None, italic=False, first_indent=True, space_after=5):
    p = doc.add_paragraph(style="Normal")
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = 1.35
    if first_indent:
        p.paragraph_format.first_line_indent = Pt(22)
    if bold_lead and text.startswith(bold_lead):
        r1 = p.add_run(bold_lead)
        set_run_font(r1, bold=True)
        r2 = p.add_run(text[len(bold_lead):])
        set_run_font(r2)
    else:
        r = p.add_run(text)
        set_run_font(r)
        r.italic = italic
    return p


def add_bullets(doc, items, level=0):
    for item in items:
        p = doc.add_paragraph(style="List Bullet" if level == 0 else "List Bullet 2")
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.line_spacing = 1.25
        r = p.add_run(item)
        set_run_font(r)


def add_numbered(doc, items):
    for number, item in enumerate(items, 1):
        p = doc.add_paragraph(style="Normal")
        p.paragraph_format.left_indent = Pt(22)
        p.paragraph_format.first_line_indent = Pt(-18)
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.line_spacing = 1.25
        r = p.add_run(f"{number}. {item}")
        set_run_font(r)


def add_heading(doc, text, level=1):
    p = doc.add_paragraph(style=f"Heading {level}")
    p.paragraph_format.keep_with_next = True
    p.paragraph_format.space_before = Pt(14 if level == 1 else 10)
    p.paragraph_format.space_after = Pt(6)
    r = p.add_run(text)
    set_run_font(
        r,
        east="微软雅黑",
        west="Arial",
        size={1: 16, 2: 13, 3: 11.5}.get(level, 11),
        bold=True,
        color=BLACK,
    )
    return p


def add_table(doc, headers, rows, widths=None, font_size=9):
    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    set_table_borders(table)
    hdr = table.rows[0]
    repeat_header(hdr)
    for i, h in enumerate(headers):
        cell = hdr.cells[i]
        set_cell_shading(cell, NAVY)
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        set_cell_margins(cell)
        if widths:
            cell.width = Inches(widths[i])
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.line_spacing = 1.08
        r = p.add_run(str(h))
        set_run_font(r, east="微软雅黑", west="Arial", size=font_size, bold=True, color=WHITE)
    for ridx, row in enumerate(rows):
        cells = table.add_row().cells
        cant_split(table.rows[-1])
        for i, value in enumerate(row):
            cell = cells[i]
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            set_cell_margins(cell)
            if widths:
                cell.width = Inches(widths[i])
            if ridx % 2 == 1:
                set_cell_shading(cell, PALE_BLUE)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if i == 0 else WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.12
            r = p.add_run(str(value))
            set_run_font(r, size=font_size)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return table


def add_label_para(doc, label, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(5)
    p.paragraph_format.line_spacing = 1.3
    r = p.add_run(label)
    set_run_font(r, bold=True)
    r2 = p.add_run(text)
    set_run_font(r2)
    return p


def add_topic(doc, idx, topic):
    topic_heading = add_heading(doc, f"{idx}  {topic['title']}", 1)
    topic_heading.paragraph_format.page_break_before = True
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(10)
    r = p.add_run(topic["english"])
    set_run_font(r, east="Arial", west="Arial", size=11, bold=True)
    r.italic = True

    add_label_para(doc, "论文定位  ", topic["position"])
    add_label_para(doc, "一句话论证  ", topic["argument"])
    add_label_para(doc, "创新边界  ", topic["boundary"])

    add_heading(doc, f"{idx}.1 研究问题与研究内容", 2)
    add_body(doc, topic["problem"])
    add_bullets(doc, topic["research_content"])

    add_heading(doc, f"{idx}.2 方法框架与模块设计", 2)
    add_body(doc, topic["method_overview"])
    add_table(
        doc,
        ["模块", "要解决的问题", "实现机制", "直接验证证据"],
        topic["modules"],
        widths=[1.25, 1.65, 2.35, 1.65],
        font_size=8.6,
    )

    add_heading(doc, f"{idx}.3 详细实现流程", 2)
    add_numbered(doc, topic["workflow"])

    add_heading(doc, f"{idx}.4 实验设计", 2)
    add_body(doc, topic["experiment_intro"])
    add_table(
        doc,
        ["编号", "实验问题", "对照或消融", "数据与条件", "输出和判定"],
        topic["experiments"],
        widths=[0.45, 1.20, 1.60, 1.78, 1.87],
        font_size=8.2,
    )

    add_heading(doc, f"{idx}.5 数据集 指标与统计规范", 2)
    add_table(doc, ["类别", "具体安排"], topic["data_metrics"], widths=[1.35, 5.55], font_size=9)

    add_heading(doc, f"{idx}.6 论文大纲", 2)
    add_numbered(doc, topic["outline"])

    add_heading(doc, f"{idx}.7 预期成果与验收闸门", 2)
    add_table(doc, ["成果", "预期交付", "进入论文结论的最低证据"], topic["outputs"], widths=[1.25, 2.85, 2.80], font_size=8.8)

    add_heading(doc, f"{idx}.8 风险 失败判据与降级路线", 2)
    add_table(doc, ["主要风险", "失败判据", "处理或降级方案"], topic["risks"], widths=[1.75, 2.20, 2.95], font_size=8.8)

    add_heading(doc, f"{idx}.9 计划安排", 2)
    add_table(doc, ["阶段", "工作内容", "阶段交付"], topic["schedule"], widths=[1.25, 3.75, 1.90], font_size=8.8)

    add_heading(doc, f"{idx}.10 文献与代码锚点", 2)
    add_bullets(doc, topic["anchors"])


topics = [
    {
        "title": "面向空间非对齐的双坐标可靠性感知双光谱反无人机检测",
        "english": "Dual-Coordinate Reliability-Aware Dual-Spectrum Anti-UAV Detection under Spatial Misalignment",
        "position": "六个题目中的核心算法主线。研究空间位置和输出坐标是否可信，不把模态退化作为主要贡献。",
        "argument": "在可见光与热红外天然不对齐的反无人机场景中，通过双坐标独立监督、空间可靠性估计和目标级坐标仲裁，使检测器在错位增强时比固定坐标融合更稳定，并明确其适用错位范围。",
        "boundary": "只处理平移、旋转、尺度差、视差和局部视场不重合造成的空间非对齐；持续过曝、热交叉和帧损由题目二研究；内部目标级特征交互由题目六研究。",
        "problem": "当前项目已经具备可见光标签与红外标签两套坐标监督，并观察到融合头在仅红外可见的困难帧上仍跟随RGB坐标。现有困难子集报告中，509帧仅红外可见样本上，融合头mAP50-95约为0.0011，而红外头约为0.5760。该结果需要在论文前重新复跑和归档，但它已把科学问题从“再设计一个融合模块”收敛为“融合结果应落在哪个坐标系，以及何时应切换坐标系”。",
        "research_content": [
            "建立双坐标检测任务定义：RGB头在可见光坐标系输出，IR头在红外坐标系输出，融合头不再默认绑定RGB标签。",
            "估计像素级或区域级空间可靠性，包括局部偏移不确定性、单模态定位置信度和目标是否位于共同视场。",
            "建立目标级跨坐标关联，将同一目标在两个坐标系中的候选框关联后进行坐标选择，而不是直接平均两个框。",
            "设计单模态保底机制：当关联不可靠或目标只在一路可见时，输出退化为健康分支的独立检测结果。",
            "建立天然非对齐加附加扰动的压力测试协议，报告最坏错位档而非只报告平均mAP。",
        ],
        "method_overview": "方法以现有dual-fusion与B1双坐标监督为起点。RGB和IR分支分别形成检测候选；共享语义层估计跨模态目标对应；空间可靠性头给出候选关联可靠度与两个坐标系的定位可靠度；坐标仲裁器据此选择RGB框、IR框或经过校正的融合框。每个模块都设置可关闭开关，保证能通过消融把增益归因到监督、可靠性或仲裁。",
        "modules": [
            ["双坐标独立检测头", "单一RGB坐标监督使融合头依赖RGB", "RGB与IR使用各自真实框监督，保留独立分类和回归头", "两路单模态上界与双头基线"],
            ["空间可靠性估计", "错位程度随位置和目标变化", "结合局部相关性、回归方差、共同视场掩码与检测置信度输出可靠性", "可靠性与注入偏移或匹配正确性的相关性"],
            ["目标级坐标关联", "两个坐标系不能直接逐像素对应", "几何邻域加语义相似度完成候选匹配，保留未匹配候选", "匹配Precision Recall及错配类型"],
            ["坐标仲裁与兜底", "错误融合会比单模态更差", "在RGB框、IR框、校正框和单模态输出之间进行可解释选择", "最坏错位档相对健康单模态的差值"],
        ],
        "workflow": [
            "冻结数据协议，重新划分训练、验证和独立测试序列，禁止Anti-UAV300同一序列跨集合泄漏。",
            "运行prepare_mm_dataset.py的dual模式，检查labels与labels_ir一一对应，并抽样可视化两套框。",
            "训练single_rgb、single_ir、单头融合、dual-fusion和B1双坐标监督五个基线，每个关键配置至少三个随机种子。",
            "用真实双谱框差统计天然偏移分布，按目标尺度、场景和共同视场状态分层，不构造不可达的理想零偏移基准。",
            "实现附加几何扰动器，在天然偏移上叠加平移、旋转与尺度变化，并保存每帧扰动真值。",
            "实现可靠性头，先单独训练或联合训练其对扰动等级、共同视场和候选匹配正确性的预测。",
            "实现目标级跨坐标候选关联，优先保证召回，再用语义相似度与负样本挖掘降低错配。",
            "实现坐标仲裁器与未匹配候选兜底路径，确保一路目标缺失时不强制融合。",
            "完成主表、错位性能曲线、可靠性校准图、困难子集结果和典型失败案例可视化。",
            "在外部数据或不同设备序列上只做最终一次复核，冻结阈值后不得针对测试集调参。",
        ],
        "experiment_intro": "实验必须证明三件事：非对齐确实伤害朴素融合；双坐标监督减少坐标偏置；可靠性仲裁在强错位和单模态可见时避免灾难性退化。",
        "experiments": [
            ["E1", "双坐标监督是否必要", "单头RGB标签 双头独立标签 B1双坐标融合监督", "同一划分和训练预算", "三路mAP及相对各自单模态基线的增益"],
            ["E2", "错位增大时性能如何变化", "早期拼接 中期Concat 固定加权 可靠性方法", "天然偏移加2 8 16像素及小角度旋转尺度扰动", "mAP Recall与下降斜率曲线"],
            ["E3", "坐标选择是否优于固定输出", "固定RGB 固定IR 置信度选择 可靠性选择 Oracle", "共同视场与部分出画分别统计", "选择准确率 定位误差 最坏档性能"],
            ["E4", "可靠性值是否可信", "不确定性单源 多源组合 去除共同视场信息", "已知扰动掩码与匹配真值", "AUROC AUPRC ECE Spearman相关"],
            ["E5", "仅一路可见时能否兜底", "完整输入 仅RGB 仅IR 屏蔽RGB融合项 屏蔽IR融合项", "困难子集only_ir与only_vis", "分支mAP 检出率 相对健康单模态差值"],
            ["E6", "结论能否跨域", "域内训练零样本外测与少量微调", "Anti-UAV410及其他可获得双谱或单模态集", "方向一致性与适用边界"],
            ["E7", "代价是否可接受", "不同候选数与可靠性头规模", "相同GPU及输入尺寸", "参数量 FLOPs 吞吐 延迟 显存"],
        ],
        "data_metrics": [
            ["训练与验证", "重建序列级互斥划分；双标签必须来自各自相机真实框；保留负样本和共同视场标记。"],
            ["主指标", "mAP50-95、mAP50、Recall、定位误差；分别在RGB坐标、IR坐标和最终输出坐标报告。"],
            ["专项指标", "最坏错位档性能、错位曲线斜率、坐标选择准确率、可靠性ECE、仅单模态可见检出率。"],
            ["统计", "关键模型至少3个随机种子，报告均值与标准差；固定阈值；对同一测试帧采用配对bootstrap置信区间。"],
            ["主文证据", "主表、错位曲线、坐标选择消融、困难子集和一组失败案例。更细扰动网格与超参数进入补充材料。"],
        ],
        "outline": [
            "引言：非对齐双谱反无人机检测的输出坐标问题与现有融合的隐含RGB锚定。",
            "相关工作：弱配准融合、可靠性估计、双头检测和跨模态候选关联。",
            "问题定义：双坐标输入、双标签、共同视场、输出坐标与可靠性变量。",
            "方法：双坐标检测、可靠性估计、目标级关联、坐标仲裁与训练目标。",
            "实验：基线、错位压力测试、可靠性校准、困难子集、跨域与复杂度。",
            "讨论：哪些错位可处理、目标出画和两路均不可见时为何失败。",
            "结论：以证据限定双坐标可靠性方法的适用边界。",
        ],
        "outputs": [
            ["算法", "双坐标可靠性感知检测模型和可关闭模块", "强错位下优于固定坐标策略且差异超过种子波动"],
            ["评测协议", "天然偏移统计与附加扰动测试集生成器", "扰动可复现并与真实标定残差范围对应"],
            ["论文图表", "主表 错位曲线 校准图 困难案例", "每个核心声明至少有一项直接实验"],
            ["代码与数据", "配置 权重 评测脚本 复现实验清单", "第三方可从固定配置复现主表"],
        ],
        "risks": [
            ["坐标关联错误", "匹配精度低且错误随错位迅速增加", "保留未匹配候选 单模态兜底 将论文收缩为双坐标监督与评测协议"],
            ["可靠性无校准", "可靠性与错位或正确性相关性弱", "改为可监督的扰动等级和共同视场预测 减少抽象可靠性定义"],
            ["与题目六重叠", "主要贡献落到候选语义注意力", "本题只保留关联用于坐标选择 语义交互性能提升留给题目六"],
            ["独立测试不足", "测试序列参与训练或调参", "重训序列级划分并增加外部复核 不沿用泄漏指标"],
        ],
        "schedule": [
            ["第1至2周", "重建划分 双标签审计 基线复跑", "数据审计表与五基线"],
            ["第3至4周", "偏移统计与压力测试协议", "天然偏移分布和退化曲线"],
            ["第5至7周", "可靠性头 目标关联 坐标仲裁", "完整模型与核心消融"],
            ["第8至9周", "困难子集 跨域 复杂度 可视化", "最终图表和失败分析"],
            ["第10周", "论文初稿与复现包", "可投稿初稿"],
        ],
        "anchors": [
            "现有代码：prepare_mm_dataset.py、train_full.py、eval_hard_subset.py、diag_fusion_attribution.py，以及dual-fusion和B1配置。",
            "现有证据：困难子集only_ir与only_vis分组、融合头归因诊断、双标签坐标监督方案。",
            "文献线索：CoDAF、CF-YOLO、LER-YOLO及未配准RGB-T检测工作；正式投稿前需逐篇核对版本和实验口径。",
        ],
    },
    {
        "title": "面向间歇性模态失效的健康状态驱动双光谱自适应融合检测",
        "english": "Health-State-Driven Adaptive Dual-Spectrum Detection under Intermittent Modality Failure",
        "position": "模态鲁棒性论文。以退化检测、健康状态估计和恢复过程为核心，不研究输出坐标选择。",
        "argument": "通过把可见光与热红外的局部退化和间歇帧损建模为连续健康状态，并用健康状态控制融合和迟滞切换，使双光谱检测在已见与未见退化下更接近健康单模态上界。",
        "boundary": "主要处理过曝、欠曝、模糊、雾、热交叉、增益漂移、噪声、坏点和突发丢帧；空间错位只作为固定干扰，不把几何对齐作为贡献。",
        "problem": "当前模型在RGB失效帧上未能切换到仍然有效的IR分支，说明融合权重并不等于模态健康度。工程值守中的失效通常不是永久断路，而是区域退化、渐变恶化、若干帧掉线后恢复。论文需要回答模型能否识别何时不应相信某一模态，以及切换后能否稳定恢复而不产生权重抖动。",
        "research_content": [
            "定义模态健康状态：全局健康度、区域健康图和时序状态三层表示。",
            "建立可复现退化算子库，同时保存退化类型、严重度、掩码和时间段真值。",
            "训练健康状态估计头，联合图像统计、特征统计和检测不确定性判断模态可用程度。",
            "用健康状态控制门控融合，并用迟滞或状态机防止权重在连续帧间抖动。",
            "建立恢复时间、最坏性能和健康单模态差距等鲁棒性指标。",
        ],
        "method_overview": "RGB和IR分支先各自输出多尺度特征与质量统计。健康估计器形成区域健康图和全局状态；门控融合器根据健康度抑制退化特征；时序平滑器用短窗口或迟滞阈值稳定切换；当一路健康度低于阈值时，检测器显式退化为另一路。训练包括局部退化增强、突发丢帧序列和健康监督。",
        "modules": [
            ["退化生成器", "真实失效类型异质且缺少标签", "参数化生成视觉和红外退化并保存真值", "退化可视化及真实样例相似性审查"],
            ["健康状态估计器", "检测置信度不能区分输入质量与模型错误", "融合亮度梯度噪声特征统计和预测不确定性", "退化掩码IoU 状态识别AUC 校准误差"],
            ["健康驱动门控", "固定融合会被坏模态拖累", "区域级权重加单模态兜底", "相对无感知融合和健康单模态的差值"],
            ["时序迟滞切换", "间歇故障会导致频繁切换", "短窗口平滑和进入退出不同阈值", "切换次数 恢复帧数 告警连续性"],
        ],
        "workflow": [
            "整理真实退化清单，按传感器机制区分RGB过曝欠曝模糊雾和IR热交叉增益漂移噪声坏点。",
            "实现退化算子与严重度分级，所有退化均保存参数、区域掩码和随机种子。",
            "构造突发丢帧和恢复序列，包括单帧、短突发和渐变恢复三种时间模式。",
            "训练完整双模态、模态dropout和固定门控基线，复跑当前only_ir与only_vis困难子集。",
            "实现健康状态估计头，先验证健康预测，再接入融合，避免检测增益掩盖健康估计失败。",
            "实现区域门控、全局门控和二者组合，保留健康单模态直接输出分支。",
            "加入时序平滑与迟滞阈值，记录每次失效的检测下降、切换时刻和恢复时间。",
            "对训练未见退化类型做零样本压力测试，检验健康度是否只记住合成算子。",
            "在真实逆光、低照或热背景片段上进行盲评，由人工质量标签辅助解释。",
            "完成退化矩阵、恢复曲线、健康图可视化、消融与失败边界。",
        ],
        "experiment_intro": "主证据不应只是正常验证集mAP提升，而应是退化类型乘严重度矩阵、突发失效时间曲线和未见退化泛化。",
        "experiments": [
            ["E1", "固定融合是否受坏模态拖累", "RGB IR 固定融合 模态dropout", "各退化类型三至五级严重度", "性能退化曲线和健康单模态差距"],
            ["E2", "健康状态能否被识别", "图像统计 特征统计 预测不确定性 组合", "退化掩码和状态真值", "IoU AUC F1 ECE"],
            ["E3", "区域健康图是否优于全局权重", "全局门控 区域门控 二者组合", "局部过曝 局部热交叉 局部遮挡", "局部目标召回和背景误检"],
            ["E4", "间歇失效后能否快速恢复", "无时序 平滑 迟滞 状态机", "不同长度帧损与恢复序列", "最低召回 恢复帧数 权重切换次数"],
            ["E5", "能否泛化到未见退化", "见过与未见过算子分组", "留一退化类型外验证", "相对无感知融合的增益及校准"],
            ["E6", "真实困难帧是否成立", "基线与健康驱动模型", "only_ir only_vis及实采质量标注片段", "分组mAP 召回 典型健康图"],
            ["E7", "额外开销是否合理", "不同健康头规模与更新频率", "统一硬件与批量", "参数量 延迟 显存"],
        ],
        "data_metrics": [
            ["退化数据", "公开双谱训练对加合成退化；真实困难帧只用于最终评测和退化逼真度审查。"],
            ["主指标", "mAP50-95、Recall、退化相对下降率、健康单模态差距和最坏条件性能。"],
            ["时序指标", "故障检测延迟、切换次数、恢复帧数、连续漏检长度。"],
            ["健康指标", "区域掩码IoU、状态分类AUC、ECE和健康度与性能下降的秩相关。"],
            ["统计", "同一干净帧派生的不同方法结果采用配对分析；随机退化固定种子并发布配置。"],
        ],
        "outline": [
            "引言：双谱值守中的部分退化与间歇失效为何不同于永久模态缺失。",
            "相关工作：模态dropout、质量感知融合、传感器故障鲁棒检测。",
            "退化与健康状态定义：空间和时间两种粒度。",
            "方法：健康估计、健康驱动门控、单模态兜底和迟滞恢复。",
            "实验：退化矩阵、健康校准、恢复曲线、未见退化、真实困难帧和开销。",
            "讨论：合成退化的真实性、两路同时失效和长期离线边界。",
            "结论：健康状态在何种条件下能降低坏模态拖累。",
        ],
        "outputs": [
            ["算法", "健康状态估计与自适应融合网络", "多种退化和严重度下稳定优于无感知融合"],
            ["退化基准", "可复现退化算子和时序失效协议", "参数与真值完整且真实样例审查通过"],
            ["分析", "健康校准图和恢复曲线", "健康度能预测性能下降而非仅复现算子标签"],
            ["论文", "鲁棒检测论文与复现配置", "未见退化和真实片段均给出边界"],
        ],
        "risks": [
            ["合成退化不真实", "模型只在已见算子上有效", "使用真实片段校准参数 留一退化验证 将结论限定为受控退化"],
            ["健康度与置信度同质", "健康头不比置信度门控更好", "增加输入统计与校准损失 或将贡献收缩为时序失效协议"],
            ["保底无法完全保证", "某些场景仍低于健康单模态", "报告软保底差距 不宣称理论保证 使用显式单模态旁路"],
            ["与题目一重叠", "主要实验转向错位", "固定几何条件 不讨论坐标选择"],
        ],
        "schedule": [
            ["第1至2周", "退化库和失效序列协议", "可视化样例和参数表"],
            ["第3至4周", "模态dropout与健康估计基线", "健康识别主表"],
            ["第5至6周", "门控融合和时序迟滞", "退化矩阵与恢复曲线"],
            ["第7至8周", "未见退化与真实困难帧", "泛化和边界结果"],
            ["第9周", "写作与复现包", "投稿初稿"],
        ],
        "anchors": [
            "现有代码：train_full.py已包含mm_modal_dropout与融合颈屏蔽参数，困难子集评测脚本可直接复用。",
            "现有证据：融合头在RGB失效帧上未切换到IR，构成健康感知研究的直接动机。",
            "文献线索：模态不完整检测、Modality Dropout和质量感知RGB-T融合；正式投稿前核对具体版本。",
        ],
    },
    {
        "title": "频率引导的多尺度动态门控双光谱无人机小目标检测",
        "english": "Frequency-Guided Multi-Scale Dynamic Gating for Dual-Spectrum Small-UAV Detection",
        "position": "最容易先形成完整初稿的弱小目标论文。贡献必须落在尺度归因和噪声控制，避免成为模块堆叠。",
        "argument": "通过分离可见光边缘纹理、红外热结构与背景高频噪声，并按目标尺度动态分配频率信息，使性能增益集中于极小和小目标，同时控制复杂背景虚警。",
        "boundary": "研究频率内容和特征尺度，不解决模态失效、坐标仲裁或开放集识别。",
        "problem": "当前项目中约四分之一目标边长低于32像素，主体分布在32至64像素。小目标的有效轮廓与压缩噪声、云层边缘和树枝纹理同属高频信息。单纯加强高频可能提高召回，也可能显著增加背景虚警。因此论文不能只证明总体mAP上升，还要回答高频对哪个尺度有效、动态门控何时抑制高频以及错误发生在什么背景。",
        "research_content": [
            "建立RGB与IR频率成分的任务化分解，而不是把两种模态使用同一频率滤波器。",
            "在P2至P5多尺度上注入频率信息，比较只在浅层、只在深层和跨尺度注入。",
            "用目标尺度、照度与模态差异驱动动态门控，避免所有位置无条件增强。",
            "把背景虚警率、尺度分桶召回和门控行为作为核心机制证据。",
            "控制新增算子为标准卷积、池化、逐元素运算或小波可展开算子，为后续部署留出空间。",
        ],
        "method_overview": "两路骨干先提取多尺度特征。频率分解模块从RGB提取边缘纹理，从IR提取局部热结构，并显式抑制孤立噪点。尺度路由器根据特征尺度和场景质量产生门控权重，将频率补偿注入P2至P4。检测头保持统一，使论文可以通过逐项移除频率分解、噪声抑制、P2分支和动态门控完成因果消融。",
        "modules": [
            ["模态特定频率分解", "RGB与IR的有效频率含义不同", "RGB强调轮廓纹理 IR强调热斑边界与局部结构", "各模态频带消融"],
            ["噪声抑制", "高频增强易放大云层树枝和传感器噪声", "局部一致性和可学习软阈值抑制孤立响应", "背景虚警率与负样本热图"],
            ["多尺度注入", "弱小目标在下采样中消失", "在P2 P3 P4分别进行受控补偿", "按目标尺寸分桶AP与Recall"],
            ["动态门控", "不同尺度和照度不应使用相同频带", "尺度照度模态差异共同生成权重", "门控分布和固定门控对照"],
        ],
        "workflow": [
            "复跑统一双流基线并记录至少三个随机种子的波动范围。",
            "统计训练、验证和测试目标尺寸，固定极小小中目标分桶，之后不因结果调整阈值。",
            "实现三类频率基线：固定高通、DCT或FFT频带、小波分解，统一参数预算比较。",
            "分别在RGB、IR和双路加入频率模块，确定模态特定分解是否必要。",
            "实现噪声抑制并建立纯背景困难集，记录每百帧虚警与虚警类型。",
            "依次在P2、P3、P4和多尺度组合注入，形成尺度位置消融。",
            "实现固定权重、通道门控、空间门控和尺度条件门控，避免一次混入多个新机制。",
            "完成二维组合消融，验证频率分解与动态门控是互补还是重复。",
            "绘制技术乘尺度的增益热图、门控权重分布和典型正负样例。",
            "在第二数据域复核尺度规律，并报告复杂度和推理速度。",
        ],
        "experiment_intro": "实验核心是机制归因。若总体指标提升但极小目标没有受益，或虚警显著增加，则题目中的小目标和频率引导均不能成立。",
        "experiments": [
            ["E1", "哪种频率表达有效", "高通 DCT FFT 小波 可学习卷积", "统一基线和参数预算", "总体及分桶AP 参数量"],
            ["E2", "哪一路模态需要频率增强", "仅RGB 仅IR 双路 共享模块", "昼间 夜间 热背景分组", "分场景增益"],
            ["E3", "注入在哪个尺度有效", "P2 P3 P4 P2加P3 多尺度", "极小小中三档目标", "分桶Recall与AP"],
            ["E4", "动态门控是否必要", "无门控 固定权重 通道 空间 尺度条件", "不同照度和目标尺寸", "门控消融和权重分布"],
            ["E5", "高频是否放大背景噪声", "有无噪声抑制", "无目标帧 云层 树枝 建筑边缘", "每百帧虚警 误检置信度"],
            ["E6", "两个模块是否真正互补", "基线 频率 门控 频率加门控", "同训练预算", "交互项增益和种子方差"],
            ["E7", "能否跨域复现", "域内与外部数据零样本", "DUT或可获得无人机数据及红外小目标集", "尺度规律是否一致"],
            ["E8", "代价是否适合后续部署", "模块规模和注入层数", "统一设备", "FLOPs 参数 延迟 显存"],
        ],
        "data_metrics": [
            ["尺寸分桶", "建议按当前统计固定为极小小于32像素、小目标32至64像素、中目标大于64像素，并同时报告COCO尺度指标。"],
            ["主指标", "mAP50-95、APsmall、Recall及三尺度分桶召回。"],
            ["虚警指标", "每百帧虚警数、纯背景平均最高置信度、典型背景类别分组。"],
            ["解释指标", "频带能量、门控权重与目标尺度或照度的关系；只作机制证据，不替代检测指标。"],
            ["统计", "差异必须超过基线多种子波动；主表报告均值与标准差。"],
        ],
        "outline": [
            "引言：双谱无人机小目标的信息损失与高频噪声矛盾。",
            "相关工作：红外小目标频域方法、多尺度检测和动态融合。",
            "方法：模态特定频率分解、噪声抑制、多尺度注入和动态门控。",
            "实验设置：尺度分桶、背景困难集、基线与公平训练协议。",
            "结果：主比较、频率消融、尺度归因、虚警、跨域和复杂度。",
            "讨论：哪些目标尺度和背景受益，何时高频无效。",
            "结论：频率信息的可用范围和部署代价。",
        ],
        "outputs": [
            ["算法", "频率引导多尺度动态门控模块", "极小或小目标增益超过随机波动且虚警可控"],
            ["机制结果", "尺度增益热图和门控行为分析", "至少两个数据域呈现同方向规律"],
            ["评测资源", "尺度分桶脚本与背景困难集", "阈值固定且可复现"],
            ["论文", "小目标检测论文和开源强基线", "模块贡献可由完整二乘二消融解释"],
        ],
        "risks": [
            ["增益只来自P2头", "频率模块移除后性能不降", "把P2作为强基线 收缩频率模块或改为分析论文"],
            ["虚警明显增加", "纯背景误检抵消召回收益", "加入软阈值与门控 限制注入层级 如实报告权衡"],
            ["模块堆叠", "无法解释频率和门控各自作用", "执行二乘二消融和反向移除 不再新增第三复杂模块"],
            ["跨域规律不一致", "第二数据集增益方向反转", "将结论限定于目标像素范围和背景类型"],
        ],
        "schedule": [
            ["第1周", "基线波动和尺度统计", "固定分桶与基线表"],
            ["第2至3周", "频率表达与模态消融", "频率主表"],
            ["第4周", "多尺度注入与P2强基线", "尺度消融"],
            ["第5周", "动态门控和噪声抑制", "组合消融与虚警表"],
            ["第6周", "跨域复核 复杂度 写作", "投稿初稿"],
        ],
        "anchors": [
            "项目依据：已有目标尺寸统计、消融清单和双流YOLO训练框架。",
            "文献线索：FD2-Net、WaveMamba、频率空间解耦和动态小目标融合工作。",
            "代码原则：优先使用可导出的标准算子，为题目四的RKNN部署保留兼容路径。",
        ],
    },
    {
        "title": "面向RK3588边缘部署的非对齐双光谱检测蒸馏量化协同优化",
        "english": "Distillation and Quantization Co-Optimization for Misaligned Dual-Spectrum Detection on RK3588",
        "position": "工程验证型论文和毕业设计部署主线。只有完成真实RK3588测试，题目才成立。",
        "argument": "通过部署约束下的跨模态知识蒸馏、量化敏感层识别和混合精度策略，在RK3588上建立非对齐双光谱检测的精度、延迟、模型大小和功耗帕累托前沿。",
        "boundary": "结论绑定RK3588、RKNN Toolkit2版本、输入尺寸和相机预处理链路；桌面GPU速度或理论FLOPs不能替代真机证据。",
        "problem": "项目已经有RKNN部署接口文件和兼容性验证方案，但deploy_rknn.py目前仍是NotImplementedError骨架。多模态模型导出实测为单个6通道输入，部署端需要完成两路预处理和拼接。自定义注意力或可变形算子可能转换失败或回退CPU，因此正确顺序是先做随机权重兼容性预检，再决定教师和学生结构，而不是训练结束后才发现模型不可部署。",
        "research_content": [
            "建立部署前算子兼容性门禁，记录ONNX算子、RKNN分配和CPU回退。",
            "设计强教师与轻量学生，蒸馏对象覆盖分类响应、定位分布和跨模态融合特征。",
            "分析PTQ、QAT与混合精度对融合层、检测头和非对齐压力测试的影响。",
            "在同一设备上测量端到端延迟、P95尾延迟、功耗、温度和长时稳定性。",
            "形成可复现的精度速度体积功耗帕累托前沿，而不是单点模型宣传。",
        ],
        "method_overview": "先用随机或短训权重导出所有候选结构，淘汰不兼容算子。教师采用前三个算法题中验证最强且可导出的模型，学生采用轻量双流结构。蒸馏同时约束检测响应、定位分布和融合特征；量化阶段先做FP16和INT8 PTQ，再根据逐层误差选择QAT或敏感层混合精度。最后把取图、预处理、6通道拼接、NPU推理和后处理纳入同一计时链。",
        "modules": [
            ["兼容性门禁", "自定义算子可能无法上NPU", "随机权重导出ONNX并检查RKNN逐层分配", "算子清单 CPU回退比例 转换日志"],
            ["跨模态蒸馏", "轻量学生易丢失模态互补和定位信息", "响应 定位 融合特征三级蒸馏", "各蒸馏项消融和教师学生差距"],
            ["量化敏感性分析", "融合与检测层对INT8误差敏感", "逐层误差和输出相似度定位敏感层", "层级误差图及精度下降"],
            ["混合精度策略", "全INT8可能精度损失过大", "敏感层保留FP16其余INT8", "帕累托前沿和真机速度"],
        ],
        "workflow": [
            "确认RK3588板卡、系统镜像、librknnrt和RKNN Toolkit2版本，建立版本锁定表。",
            "补全deploy_rknn.py的ONNX导出、校准集构建、RKNN转换和benchmark接口。",
            "对候选教师与学生结构用随机权重导出，检查是否展开为标准ONNX算子。",
            "执行RKNN转换并检查逐层NPU分配；核心融合算子回退CPU的结构立即淘汰或替换。",
            "构建200至500帧校准集，按昼夜、背景、目标尺寸和两模态质量分层抽样。",
            "训练教师、无蒸馏学生、响应蒸馏学生和多级蒸馏学生，统一训练预算。",
            "生成FP32、FP16、INT8 PTQ和INT8 QAT模型，必要时配置敏感层混合精度。",
            "分别在干净集、非对齐压力集和困难子集上评估量化前后精度。",
            "在板端测量纯NPU推理、含后处理延迟和采集到输出的端到端延迟。",
            "测量功耗、温度、NPU占用和72小时稳定性，记录崩溃、内存增长和帧率漂移。",
            "绘制精度延迟体积功耗帕累托图，选择主模型与低功耗备选模型。",
        ],
        "experiment_intro": "工程论文的核心不是模型最小，而是每个压缩步骤的收益和代价可解释，并且真机端到端数据能复现。",
        "experiments": [
            ["E1", "候选结构能否完整上NPU", "教师 学生 不同融合模块", "随机权重ONNX和RKNN转换", "转换成功率 CPU回退 算子清单"],
            ["E2", "蒸馏是否弥补轻量化损失", "无蒸馏 响应 定位 特征 组合", "同训练集和预算", "mAP差距及学生复杂度"],
            ["E3", "量化方案如何影响精度", "FP32 FP16 PTQ INT8 QAT INT8 混合精度", "固定校准集", "总体和分场景mAP下降"],
            ["E4", "非对齐是否更怕量化", "干净集与错位档", "天然偏移加附加扰动", "量化前后曲线斜率"],
            ["E5", "校准集组成是否关键", "随机抽样 分层抽样 昼间偏置 夜间偏置", "相同样本数", "量化精度与最坏场景"],
            ["E6", "板端是否实时", "不同输入尺寸和精度模式", "RK3588真机", "NPU延迟 P95 FPS 端到端延迟"],
            ["E7", "系统能否长期运行", "短测与72小时运行", "真实视频或循环回放", "功耗 温度 内存 帧率漂移"],
            ["E8", "帕累托最优点在哪里", "所有模型统一汇总", "同设备同工具链", "精度延迟体积功耗前沿"],
        ],
        "data_metrics": [
            ["精度数据", "重建的独立测试集、非对齐压力集、only_ir和only_vis困难集。"],
            ["校准数据", "200至500帧分层校准，不能与独立测试帧重叠。"],
            ["部署指标", "模型大小、NPU推理延迟、P95端到端延迟、FPS、NPU占用、功耗、温度、内存。"],
            ["精度指标", "mAP50-95、Recall、量化损失、非对齐曲线斜率和最坏困难集性能。"],
            ["统计", "延迟预热后重复测量并报告中位数与P95；环境温度、频率策略和线程数必须记录。"],
        ],
        "outline": [
            "引言：双谱模型的端侧部署缺口和非对齐信息在压缩中的脆弱性。",
            "系统约束与工具链：RK3588、RKNN版本、6通道输入和算子门禁。",
            "方法：轻量学生、跨模态蒸馏、敏感性分析和混合精度量化。",
            "实验设置：数据、校准集、真机测量与公平计时。",
            "结果：蒸馏、量化、非对齐鲁棒、校准集、延迟功耗和稳定性。",
            "讨论：工具链绑定、CPU回退和不同板卡不可直接外推。",
            "结论：给出可复现的帕累托选择而非单一最快模型。",
        ],
        "outputs": [
            ["部署代码", "完整ONNX到RKNN和板端推理流程", "无核心算子CPU回退且日志可审计"],
            ["模型", "教师 学生 FP16 INT8和混合精度权重", "至少一个点进入可接受精度速度前沿"],
            ["真机证据", "延迟 功耗 温度 稳定性原始记录", "测试协议固定且可重复"],
            ["论文", "部署优化论文和复现包", "主结论全部由真实RK3588实验支撑"],
        ],
        "risks": [
            ["部署脚本尚未实现", "无法完成转换或板端推理", "把脚本补全列为第一个里程碑 未完成前不承诺投稿"],
            ["算子回退CPU", "核心融合层未分配到NPU", "改用标准卷积乘加门控 保留复杂教师但简化学生"],
            ["INT8掉点过大", "关键困难集损失不可接受", "QAT 混合精度 敏感层回退FP16"],
            ["真机不可获得", "只能报告桌面结果", "暂缓此题 不把FLOPs包装成边缘部署证据"],
        ],
        "schedule": [
            ["第1周", "工具链锁定和脚本补全", "可转换最小基线"],
            ["第2周", "所有候选结构兼容性预检", "算子与回退报告"],
            ["第3至5周", "教师学生训练和蒸馏", "蒸馏消融"],
            ["第6至7周", "PTQ QAT 混合精度", "量化敏感性结果"],
            ["第8至9周", "真机延迟功耗稳定性", "帕累托前沿"],
            ["第10周", "写作与复现整理", "投稿初稿"],
        ],
        "anchors": [
            "现有代码：deploy_rknn.py为接口骨架，当前需补全；RKNN兼容性验证方法文档给出执行顺序。",
            "现有工程事实：多模态ONNX可能导出为单个6通道输入，部署端需完成双路预处理和通道拼接。",
            "文献线索：多模态知识蒸馏、轻量无人机检测和网络量化；必须以真机协议为最终依据。",
        ],
    },
    {
        "title": "基于双光谱外观热斑轨迹证据融合的鸟类无人机开放集识别",
        "english": "Open-Set Bird-Drone Recognition via Dual-Spectral Appearance Thermal-Spot and Trajectory Evidence Fusion",
        "position": "场景应用型论文。与前四题差异最大，但数据准备风险最高，适合检测与跟踪稳定后开展。",
        "argument": "通过把可见光外观、红外热斑结构和短时运动轨迹作为可分解证据，并对未知飞行目标进行置信校准与拒识，降低鸟类被误报为无人机的风险。",
        "boundary": "研究检测后的目标级分类与开放集拒识；不承担初始目标检测、云台扫描或绝对距离估计。",
        "problem": "远距离鸟类和小型无人机在单帧中都可能只占少量像素。鸟类滑翔时没有明显扑翼，无人机在逆光中也可能失去结构纹理。单一外观分类很容易输出过高置信度。双光谱热斑和轨迹可提供互补证据，但论文必须处理缺失模态、跟踪断裂和训练集外飞行目标，而不能只在封闭的鸟和无人机二分类上取得高准确率。",
        "research_content": [
            "建立鸟类、无人机、已知干扰和未知飞行目标的数据划分与开放集协议。",
            "从检测框序列提取可见光外观、红外热斑分布和轨迹运动三类证据。",
            "设计质量感知证据融合，在单帧模糊、热斑缺失或轨迹过短时降低相应证据权重。",
            "建立开放集得分和校准机制，允许输出未知而不是强制鸟或无人机。",
            "分析不同距离、目标像素、运动状态和跟踪噪声下的误判来源。",
        ],
        "method_overview": "检测器与跟踪器先生成目标轨迹。外观编码器处理RGB裁剪，热结构编码器处理IR裁剪并提取热斑数量、分布和时序稳定性，轨迹编码器处理位移、速度、曲率和周期性。质量估计器为三类证据分配可靠度，证据融合头输出鸟、无人机、已知干扰和未知分数。开放集模块采用能量、原型距离或证据不确定性，并通过独立验证集校准阈值。",
        "modules": [
            ["外观证据", "远距离单帧纹理弱", "短时多帧聚合RGB轮廓纹理", "外观单独分类和尺寸分组"],
            ["热斑证据", "鸟与无人机热结构不同但分辨率有限", "热斑拓扑 局部温差和稳定性编码", "热斑可见与不可见分组"],
            ["轨迹证据", "滑翔鸟与匀速无人机仍可能相似", "短时速度 曲率 周期和机动模式", "不同窗口与跟踪噪声实验"],
            ["开放集证据融合", "封闭分类会强制未知目标归类", "质量加权证据和未知拒识得分", "AUROC OSCR FPR95和校准"],
        ],
        "workflow": [
            "制定类别本体，明确鸟、旋翼无人机、固定翼无人机、有人机、昆虫、气球和远处未知点目标的归属。",
            "盘点现有连续视频与同步双谱轨迹，若数据不足，先停止模型设计并补采或引入公开鸟机数据。",
            "按视频序列和拍摄地点划分训练验证测试，避免同一轨迹的相邻帧跨集合泄漏。",
            "运行检测和跟踪生成轨迹，人工修正轨迹ID并标记遮挡、滑翔、扑翼、悬停和快速穿越。",
            "构建RGB裁剪、IR裁剪、热斑特征和轨迹特征缓存，统一时间戳和窗口长度。",
            "训练外观单模态、热斑单模态和轨迹单模态基线，先确认每类证据具备独立信息。",
            "实现质量感知证据融合，加入模态缺失和跟踪断裂训练增强。",
            "定义已知类与未知类划分，比较能量、最大softmax、原型距离和证据网络。",
            "在跨地点或跨设备测试上冻结阈值，报告开放集曲线和鸟类虚警。",
            "分析滑翔鸟、微型无人机、远距离热斑和轨迹过短等失败案例。",
        ],
        "experiment_intro": "本题的决定性指标是鸟类虚警和未知拒识，不是封闭集准确率。数据按轨迹划分是可信度前提。",
        "experiments": [
            ["E1", "三类证据各自贡献什么", "外观 热斑 轨迹及两两组合 全组合", "同一轨迹划分", "封闭分类和鸟类虚警"],
            ["E2", "热斑拓扑是否优于普通IR特征", "IR CNN 热斑统计 热斑拓扑 组合", "不同目标像素和热对比", "分组F1和误判"],
            ["E3", "多长轨迹足够", "1 4 8 16 32帧窗口", "滑翔 扑翼 悬停 穿越", "识别率 延迟 覆盖率"],
            ["E4", "未知目标能否被拒识", "MSP 能量 原型距离 证据方法", "已知未知独立划分", "AUROC OSCR FPR95"],
            ["E5", "概率是否校准", "无校准 温度缩放 类别条件校准", "域内与跨域", "ECE Brier和风险覆盖曲线"],
            ["E6", "证据缺失时是否稳定", "缺RGB 缺IR 轨迹断裂 多项缺失", "受控屏蔽", "性能下降和未知误报"],
            ["E7", "跟踪误差影响多大", "真值轨迹 自动轨迹 加噪轨迹", "ID切换和定位噪声", "性能敏感性"],
            ["E8", "能否跨场景", "域内与跨地点跨设备", "冻结阈值", "鸟类虚警和未知召回"],
        ],
        "data_metrics": [
            ["数据前提", "必须有连续视频、轨迹ID、鸟与无人机困难样本以及至少一种训练未见目标。若缺少则此题暂不启动。"],
            ["划分", "按完整轨迹、拍摄日期和地点划分，禁止帧级随机划分。"],
            ["主指标", "鸟类虚警率、无人机召回、宏平均F1、AUROC、OSCR、FPR95。"],
            ["校准", "ECE、Brier分数、风险覆盖曲线和不同阈值下的告警负担。"],
            ["分层", "按目标像素、运动状态、轨迹长度、昼夜、热对比和跟踪质量报告。"],
        ],
        "outline": [
            "引言：鸟机混淆为何需要多源时序证据和未知拒识。",
            "数据与协议：类别本体、轨迹划分、未知类和困难子集。",
            "方法：外观、热斑、轨迹编码、质量估计和开放集融合。",
            "实验设置：跟踪器、窗口、基线、开放集指标和校准。",
            "结果：证据消融、轨迹长度、未知拒识、缺失证据、跨域和失败案例。",
            "讨论：开放集定义、数据偏差和实际告警阈值的边界。",
            "结论：多源证据在哪些条件下降低鸟类虚警。",
        ],
        "outputs": [
            ["数据", "轨迹级鸟机困难集和开放集划分", "类别和轨迹审计完整 无帧泄漏"],
            ["算法", "质量感知三证据融合与未知拒识", "跨域鸟类虚警下降且未知AUROC优于基础方法"],
            ["分析", "运动状态和证据缺失敏感性", "失败案例与警戒边界可解释"],
            ["论文", "开放集鸟机识别论文", "主结论由轨迹级独立测试支撑"],
        ],
        "risks": [
            ["连续双谱鸟类数据不足", "无法形成轨迹级独立测试", "优先补采或使用公开单模态数据预训练 此题延后而非用帧复制伪造"],
            ["未知类定义随意", "不同划分结论波动大", "预注册多种未知划分并报告均值"],
            ["跟踪错误主导结果", "自动轨迹远差于真值轨迹", "联合报告两种上限下限 将跟踪质量作为边界"],
            ["与系统论文混杂", "加入云台和告警状态机", "本题只到目标轨迹识别输出 系统闭环另文处理"],
        ],
        "schedule": [
            ["第1至2周", "数据盘点和类别本体", "可行性闸门报告"],
            ["第3至5周", "轨迹标注与三类证据基线", "轨迹级数据集和基线"],
            ["第6至7周", "证据融合与开放集方法", "主模型和消融"],
            ["第8至9周", "跨域 校准 缺失证据", "最终图表"],
            ["第10周", "写作与数据说明", "投稿初稿"],
        ],
        "anchors": [
            "第14节路线：外观、热斑和轨迹联合的鸟机开放集判别。",
            "可复用资产：检测器输出、未来跟踪轨迹和双光谱裁剪；当前是否有足量鸟类连续视频必须先核验。",
            "文献线索：Drone-vs-Bird挑战、基于轨迹的鸟机分类和视觉反无人机跟踪。",
        ],
    },
    {
        "title": "无需像素级配准的目标级跨模态语义交互双光谱小目标检测",
        "english": "Registration-Free Object-Level Cross-Modal Semantic Interaction for Dual-Spectrum Small-Object Detection",
        "position": "高创新高风险算法论文。研究内部目标实例如何交互，不决定最终输出坐标。",
        "argument": "通过在候选目标或查询层建立跨模态语义对应，并保留未匹配目标的单模态旁路，使模型在像素错位下利用跨模态语义而不依赖稠密配准。",
        "boundary": "不宣称完成图像配准；输出坐标可固定为主坐标或沿用双头设置。与题目一的区别是本题关注特征交互和匹配质量，不关注坐标仲裁。",
        "problem": "像素级融合假设同一位置对应同一目标，在天然视差、焦距差和局部出画时不成立。目标级语义交互把对象从像素位置提升为实例候选，但小无人机可能只有十几像素，两个模态中的外观差异大，候选匹配本身可能成为新瓶颈。论文必须先证明匹配可用，再证明语义交互能提升检测；不能把匹配错误隐藏在最终mAP中。",
        "research_content": [
            "双流独立产生高召回候选或目标查询，保留低置信小目标。",
            "融合几何邻域、类别语义和局部特征相似度完成跨模态目标匹配。",
            "对匹配目标进行稀疏跨注意力，对未匹配目标保留单模态旁路。",
            "通过难负样本和循环一致性抑制匹配到云层、鸟类或另一目标。",
            "系统评估匹配精度、检测增益、错位鲁棒性和候选数量带来的复杂度。",
        ],
        "method_overview": "RGB与IR检测分支产生top-k候选和ROI特征。匹配器在允许的几何邻域内计算语义相似度并进行双向一致匹配；交互器仅在匹配对之间执行跨注意力或消息传递；未匹配候选不被强制融合。小目标保护分支使用高分辨率ROI或题目三的轻量频率特征，但不把频率模块作为本论文主贡献。",
        "modules": [
            ["高召回候选生成", "小目标低置信易在交互前被删除", "低阈值top-k候选与P2 ROI", "候选召回与数量"],
            ["目标级匹配", "像素位置不对应", "几何松弛加语义相似度加双向一致", "匹配Precision Recall F1"],
            ["稀疏跨模态交互", "全局注意力开销大且易混入背景", "只在匹配目标间交换信息", "相对无交互和稠密交互的增益"],
            ["未匹配旁路", "目标可能只在单模态出现", "保留独立候选并抑制强制融合", "部分出画和模态缺失结果"],
        ],
        "workflow": [
            "确定输出坐标和标签协议，避免在本题中同时引入坐标仲裁创新。",
            "训练双流独立检测基线，导出候选框、分类分数和多尺度ROI特征。",
            "利用双标签和序列信息构造候选匹配真值，包括正匹配、未匹配和难负样本。",
            "评估几何最近邻、IoU映射、特征余弦和可学习匹配器，先通过匹配闸门。",
            "实现双向一致与负样本挖掘，统计错配到背景、邻近鸟类和重复目标的比例。",
            "加入稀疏跨注意力或消息传递，限制候选数和邻域，记录复杂度。",
            "实现未匹配旁路与单模态可见训练样本，防止强制配对。",
            "在天然偏移和附加偏移下比较像素级、可变形和目标级方法。",
            "按目标尺寸和候选匹配正确性分组，验证检测增益是否来自正确交互。",
            "完成跨数据复核、可视化和失败边界。",
        ],
        "experiment_intro": "本题设两个闸门。第一闸门是候选匹配达到可用精度；第二闸门是正确匹配带来的检测增益超过种子波动。任一不通过都不应继续堆叠注意力模块。",
        "experiments": [
            ["E1", "候选能否覆盖真实小目标", "不同阈值 top-k P2开关", "按目标尺寸分层", "候选Recall 数量和延迟"],
            ["E2", "哪种匹配机制可靠", "几何 IoU 语义 学习匹配 组合", "双标签匹配真值", "匹配Precision Recall F1"],
            ["E3", "目标级交互是否优于像素融合", "Concat 像素注意力 可变形 目标级", "天然与附加偏移", "mAP 曲线斜率"],
            ["E4", "正确匹配是否导致增益", "真实匹配 预测匹配 随机匹配 Oracle", "同一模型", "检测增益上限和错配损失"],
            ["E5", "未匹配旁路是否必要", "强制匹配与旁路", "部分出画 仅一路可见", "Recall和误检"],
            ["E6", "小目标是否真正受益", "按小于32 32至64 大于64像素", "固定分桶", "分桶AP Recall"],
            ["E7", "候选交互代价多大", "top-k 邻域数 层数", "统一GPU", "复杂度 延迟 显存"],
            ["E8", "能否跨域", "域内和外部复核", "可获得未配准双谱集", "匹配与检测方向一致性"],
        ],
        "data_metrics": [
            ["训练数据", "双标签配对帧，并为目标匹配建立可审计真值。部分出画目标必须标为未匹配而非负样本。"],
            ["匹配指标", "候选召回、匹配Precision、Recall、F1和错配类型。"],
            ["检测指标", "mAP50-95、APsmall、Recall、错位曲线斜率和单模态可见性能。"],
            ["效率指标", "候选数、注意力配对数、延迟、显存和参数量。"],
            ["统计", "匹配阈值只在验证集确定；检测主比较至少三个随机种子。"],
        ],
        "outline": [
            "引言：像素对应假设在非配准双谱小目标中为何失效。",
            "相关工作：像素对齐、可变形交互、目标查询和跨模态匹配。",
            "问题定义：匹配目标、未匹配目标和输出坐标边界。",
            "方法：候选生成、目标匹配、稀疏语义交互和单模态旁路。",
            "实验：匹配闸门、检测比较、Oracle分析、错位、小目标、复杂度和跨域。",
            "讨论：候选质量上限、两模态语义差异和密集多目标场景。",
            "结论：目标级交互相对像素融合的适用条件。",
        ],
        "outputs": [
            ["算法", "目标级匹配与稀疏跨模态交互模型", "预测匹配接近Oracle趋势并提升强错位检测"],
            ["匹配基准", "候选匹配真值和评测脚本", "未匹配与难负样本定义清晰"],
            ["分析", "Oracle 随机匹配和错配损失研究", "能解释交互何时有效或有害"],
            ["论文", "非配准目标级交互论文", "匹配指标和检测指标同时成立"],
        ],
        "risks": [
            ["候选匹配闸门失败", "F1低且错位后崩溃", "提高候选召回 缩小论文为匹配分析 或停止该方向"],
            ["注意力增益来自参数量", "等参数控制后差异消失", "加入等参数Concat和MLP对照"],
            ["与题目一重叠", "同时强调坐标选择", "固定输出坐标 将坐标仲裁完全留给题目一"],
            ["计算量过高", "候选增多后无法实时", "top-k 稀疏邻域和共享投影 限定目标稀疏场景"],
        ],
        "schedule": [
            ["第1至2周", "候选导出和匹配真值", "匹配数据集"],
            ["第3周", "匹配基线和闸门判断", "匹配主表"],
            ["第4至6周", "稀疏交互和旁路", "检测消融"],
            ["第7周", "错位 尺度 Oracle和效率", "核心图表"],
            ["第8周", "跨域与写作", "投稿初稿"],
        ],
        "anchors": [
            "项目基础：双头检测、双标签、困难子集和非对齐增强均可复用。",
            "与其他题目关系：题目三可提供轻量小目标特征，但不应直接把其完整模块重复作为新贡献。",
            "文献线索：目标级跨模态检测、DAMSDet、ICAFusion和未配准RGB-T小目标工作。",
        ],
    },
]


def build_document():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    doc = Document()
    sec = doc.sections[0]
    sec.page_width = Inches(8.5)
    sec.page_height = Inches(11)
    sec.top_margin = Inches(0.72)
    sec.bottom_margin = Inches(0.70)
    sec.left_margin = Inches(0.78)
    sec.right_margin = Inches(0.78)

    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Times New Roman"
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    normal.font.size = Pt(11)
    normal.font.color.rgb = RGBColor(0, 0, 0)
    normal.paragraph_format.line_spacing = 1.35
    normal.paragraph_format.space_after = Pt(5)

    for level in (1, 2, 3):
        st = styles[f"Heading {level}"]
        st.font.name = "Arial"
        st._element.rPr.rFonts.set(qn("w:eastAsia"), "微软雅黑")
        st.font.color.rgb = RGBColor(0, 0, 0)
        st.font.bold = True
        st.paragraph_format.keep_with_next = True

    title_style = styles["Title"]
    title_style.font.name = "Arial"
    title_style._element.rPr.rFonts.set(qn("w:eastAsia"), "微软雅黑")
    title_style.font.color.rgb = RGBColor(0, 0, 0)
    title_style.font.size = Pt(23)
    title_style.font.bold = True

    footer = sec.footer
    add_page_number(footer.paragraphs[0])

    # Cover
    p = doc.add_paragraph(style="Normal")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(90)
    p.paragraph_format.space_after = Pt(22)
    r = p.add_run("六个SCI二区小论文选题详细研究与实验方案")
    set_run_font(r, east="微软雅黑", west="Arial", size=23, bold=True)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(10)
    r = p.add_run("双光谱反无人机检测与边缘部署项目")
    set_run_font(r, east="微软雅黑", west="Arial", size=15, bold=True)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(36)
    r = p.add_run("研究内容  方法框架  实验矩阵  实现流程  预期成果  风险与计划")
    set_run_font(r, east="宋体", west="Times New Roman", size=11)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(150)
    r = p.add_run("2026年10月9日")
    set_run_font(r, size=11)

    doc.add_page_break()
    add_heading(doc, "使用说明与核心结论", 1)
    add_body(doc, "本方案把当前项目创新点、附件子方向报告与文献综述第14节的论文路线合并为六个相互区分的研究计划。每个计划均以一个核心科学问题为中心，给出可执行方法、完整实验矩阵、验收闸门和失败后的降级路线。文中所有数值目标均表示待验证的验收条件，不表示实验已经取得相应结果。")
    add_body(doc, "总体建议是以题目三建立强小目标基线，以题目一和题目二形成空间可靠性与模态可靠性两条核心算法论文，以题目四承接真实边缘部署。题目六适合作为高创新储备；题目五需要连续鸟机视频和开放集数据，数据闸门未通过前不宜投入主要训练资源。")

    add_table(
        doc,
        ["题目", "核心问题", "当前基础", "主要新增工作", "建议优先级"],
        [
            ["一 双坐标可靠性", "空间位置与输出坐标是否可信", "双标签 双头 困难子集 归因诊断", "可靠性估计 坐标仲裁 错位协议", "重点主论文"],
            ["二 健康状态融合", "当前模态是否健康", "模态dropout 分组评测", "退化库 健康头 时序恢复", "稳健主论文"],
            ["三 频率多尺度门控", "弱小目标细节是否保留", "目标尺度统计 双流基线", "频率分解 噪声控制 尺度归因", "最快启动"],
            ["四 RK3588部署", "压缩量化后能否真机实时", "兼容性方案 部署接口骨架", "补全工具链 蒸馏 量化 真机", "毕业主线"],
            ["五 鸟机开放集", "鸟 无人机与未知目标如何区分", "系统需求与文献路线", "连续视频 轨迹标注 开放集协议", "数据后置"],
            ["六 目标级语义交互", "非配准下如何建立实例对应", "双头 双标签 非对齐增强", "匹配真值 稀疏交互 未匹配旁路", "创新储备"],
        ],
        widths=[1.25, 1.65, 1.65, 1.65, 0.72],
        font_size=8.2,
    )

    add_heading(doc, "项目现状与必须修正的前置条件", 1)
    add_bullets(doc, [
        "已核实存在的代码包括preprocess.py、finetune.py、prepare_mm_dataset.py、train_full.py、eval_hard_subset.py、diag_fusion_attribution.py和deploy_rknn.py。",
        "prepare_mm_dataset.py已支持dual模式，可同时生成RGB标签目录labels和IR标签目录labels_ir；train_full.py已包含双头、B1双坐标监督、模态dropout与融合颈屏蔽参数。",
        "现有困难子集与归因报告显示融合头在RGB失效时没有有效切换至IR。该结论具备直接研究价值，但论文前必须用最终代码、固定权重和独立测试集复跑。",
        "现有材料指出Anti-UAV300全部序列曾进入训练、验证或测试流程，因此当前指标不能替代真正独立测试。所有六题都应先重建序列级互斥划分。",
        "deploy_rknn.py目前是带NotImplementedError的接口骨架。题目四的第一项工作不是训练，而是补全导出、校准、转换和真机benchmark。",
    ])

    add_heading(doc, "统一术语", 1)
    add_table(doc, ["规范术语", "定义与使用规则", "避免混用"], [
        ["RGB 可见光", "提供外观 轮廓 纹理和背景", "彩色 可见波段随意切换"],
        ["IR 热红外", "提供热辐射对比与热斑结构", "红外 热像 热图未定义混用"],
        ["双光谱", "同一场景的RGB与IR观测", "多模态若未说明具体模态"],
        ["空间可靠性", "位置 对齐 共同视场和坐标输出可信度", "与模态健康度等同"],
        ["模态健康度", "某一传感器在当前区域或时间是否提供有效信息", "与检测置信度等同"],
        ["目标级交互", "候选框 查询或实例特征之间的跨模态信息交换", "像素级配准"],
        ["小目标", "必须报告像素尺度或固定分桶", "只用远距离 弱小等形容词"],
    ], widths=[1.35, 3.95, 1.60], font_size=9)

    add_heading(doc, "六题共用实验纪律", 1)
    add_numbered(doc, [
        "数据按完整视频序列、采集日期或设备划分，禁止相邻帧跨训练、验证和测试集合。",
        "测试集与困难子集在训练前冻结。阈值、退化档位和尺度分桶只在训练或验证数据上确定。",
        "关键模型至少运行三个随机种子，报告均值与标准差。差异未超过基线波动时，不写成有效提升。",
        "所有比较固定输入尺寸、训练轮数、数据增强、后处理阈值和评测脚本。增加训练预算必须设置等预算对照。",
        "双头结果按RGB、IR和融合分支分别报告，并明确各自标签坐标系。不同坐标标签下的mAP不能直接互相比较。",
        "主文保留核心比较、必要消融、结论改变型困难条件和失败边界；完整超参数网格与次要稳健性检查进入补充材料。",
        "不以单次最好权重代替统计结论，不以桌面GPU速度代替RK3588真机结果，不以合成退化代替全部真实退化证据。",
    ])

    add_heading(doc, "目录", 1)
    for i, t in enumerate(topics, 1):
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.line_spacing = 1.0
        r = p.add_run(f"{i}  {t['title']}")
        set_run_font(r, size=9.5, bold=True)
    for item in ["七 六题之间的共享关系与重复发表边界", "八 推荐执行顺序与阶段决策", "九 统一交付清单"]:
        p = doc.add_paragraph(style="Normal")
        p.paragraph_format.left_indent = Pt(12)
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.line_spacing = 1.0
        r = p.add_run(f"• {item}")
        set_run_font(r, size=9.5)

    for i, topic in enumerate(topics, 1):
        add_topic(doc, i, topic)

    doc.add_page_break()
    add_heading(doc, "七 六题之间的共享关系与重复发表边界", 1)
    add_body(doc, "六题可以共享数据清洗、双标签生成、双流骨干、训练框架和评测脚本，但核心问题、主指标与关键消融必须分开。共享一个预训练权重并不自动构成重复发表；重复使用同一结果表、把同一模块在不同论文中反复宣称为主要创新，或把同一困难集结论拆分发表，则存在明显风险。")
    add_table(doc, ["题目", "独占核心贡献", "主指标", "不得重复宣称"], [
        ["一", "双坐标监督 空间可靠性 坐标仲裁", "错位曲线 最坏档 坐标选择", "题目六的语义交互模块"],
        ["二", "连续健康状态 失效恢复 迟滞切换", "退化矩阵 恢复时间 健康校准", "题目一的空间坐标可靠性"],
        ["三", "频率内容 尺度路由 噪声控制", "分桶AP 背景虚警", "把P2头本身包装成频率创新"],
        ["四", "部署约束蒸馏 量化敏感性 真机前沿", "延迟 功耗 体积 精度", "前三题完整算法再次作为部署创新"],
        ["五", "外观热斑轨迹证据与未知拒识", "鸟类虚警 OSCR AUROC", "检测器本身的mAP改进"],
        ["六", "目标匹配 稀疏交互 未匹配旁路", "匹配F1 错位检测增益", "题目一的坐标仲裁"],
    ], widths=[0.55, 2.25, 2.00, 2.10], font_size=8.6)

    add_heading(doc, "八 推荐执行顺序与阶段决策", 1)
    add_table(doc, ["阶段", "优先任务", "继续条件", "停止或转向条件"], [
        ["阶段一", "题目三统一基线和尺度实验", "分桶结果稳定且虚警可控", "增益不超过种子波动则只保留强基线"],
        ["阶段二", "题目一双坐标监督和题目二退化协议", "困难子集和错位或退化曲线成立", "数据泄漏未解决则不进入论文写作"],
        ["阶段三", "题目四兼容性预检和脚本补全", "核心算子上NPU且可真机测试", "无板卡或核心算子回退则暂缓"],
        ["阶段四", "题目六候选匹配闸门", "匹配F1与检测Oracle分析支持继续", "匹配失败则不堆叠交互模块"],
        ["阶段五", "题目五轨迹级数据闸门", "连续鸟机视频和未知类足够", "数据不足则延后"],
    ], widths=[1.00, 2.15, 2.05, 1.70], font_size=8.7)
    add_body(doc, "建议先后顺序为题目三、题目一、题目二、题目四、题目六、题目五。若毕业设计必须突出系统落地，题目四可以与题目一或题目三并行，但投稿时仍应拆成算法论文与部署论文。")

    add_heading(doc, "九 统一交付清单", 1)
    add_bullets(doc, [
        "数据版本说明、序列级划分清单、双标签审计结果和数据许可记录。",
        "每个实验的配置文件、随机种子、代码版本、权重、日志和原始评测JSON。",
        "主表对应的自动汇总脚本，禁止手工复制最优数字。",
        "模型参数量、FLOPs、GPU延迟以及适用时的RK3588真机数据。",
        "至少一组成功案例、一组典型失败案例和一份结论适用边界。",
        "论文主文证据链、补充材料清单和声明到实验的逐项映射。",
    ])
    add_body(doc, "最终选题应服从实验闸门。若某条路线的关键证据未成立，应缩小主张、转为评测或分析论文，或者停止该方向，而不是继续叠加模块掩盖问题。")

    doc.core_properties.title = "六个SCI二区小论文选题详细研究与实验方案"
    doc.core_properties.subject = "双光谱反无人机检测子方向选题"
    doc.core_properties.author = "双光谱毕设项目组"
    doc.core_properties.keywords = "双光谱 反无人机 非对齐 模态失效 小目标 RK3588 开放集"
    doc.save(OUT_FILE)
    return OUT_FILE


if __name__ == "__main__":
    path = build_document()
    print(path)
