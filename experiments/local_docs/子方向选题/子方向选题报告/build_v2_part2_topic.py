# -*- coding: utf-8 -*-
"""add_topic：把每个方向的深化内容写入十个章节。"""

from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

from build_v2_part1 import (
    add_heading, add_body, add_bullets, add_numbered, add_table,
    add_label_para, add_callout, set_run_font,
    SOFT_YELLOW, SOFT_GREEN, SOFT_RED, BLACK,
)


def add_topic(doc, idx, topic):
    topic_heading = add_heading(doc, f"{idx}  {topic['title']}", 1)
    topic_heading.paragraph_format.page_break_before = True

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(4)
    r = p.add_run(topic["english"])
    set_run_font(r, east="Arial", west="Arial", size=11, bold=True)
    r.italic = True

    # 元信息行：族 / 首稿周期 / 综合可行性
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(10)
    r = p.add_run(f"所属族：{topic['family']}　|　首稿周期：{topic['cycle']}　|　综合可行性：{topic['stars']}")
    set_run_font(r, east="微软雅黑", west="Arial", size=9.5, color="444444")

    add_label_para(doc, "论文定位　", topic["position"])
    add_label_para(doc, "一句话论证　", topic["argument"])
    add_label_para(doc, "创新边界　", topic["boundary"])

    # ---------- .1 研究问题与研究内容 ----------
    add_heading(doc, f"{idx}.1 研究问题与研究内容", 2)
    add_label_para(doc, "科学问题　", topic["science_question"])
    add_body(doc, topic["problem"])
    add_body(doc, "围绕该问题，本研究的具体内容分解为以下各项：", first_indent=False, space_after=4)
    add_bullets(doc, topic["research_content"])

    # ---------- .2 方法框架与模块设计 ----------
    add_heading(doc, f"{idx}.2 方法框架与模块设计", 2)
    add_body(doc, topic["method_overview"])
    add_table(
        doc,
        ["模块", "要解决的问题", "实现机制", "直接验证证据"],
        topic["modules"],
        widths=[1.20, 1.62, 2.33, 1.65],
        font_size=8.5,
    )
    if topic.get("framework_diagram"):
        add_label_para(doc, "数据流　", topic["framework_diagram"])

    # ---------- .3 详细实现流程 ----------
    add_heading(doc, f"{idx}.3 详细实现流程", 2)
    add_body(doc, topic["workflow_intro"])
    add_numbered(doc, topic["workflow"])

    # ---------- .4 实验设计（重点） ----------
    add_heading(doc, f"{idx}.4 实验设计", 2)
    add_body(doc, topic["experiment_intro"])

    # .4.0 实验总体框架与三问
    add_label_para(doc, "实验总体目标　", topic["exp_goal"])
    add_body(doc, topic["exp_logic"], first_indent=False, space_after=4)
    add_label_para(doc, "论证链条　", topic["exp_chain"])
    add_label_para(doc, "统一实验条件　", topic["exp_conditions"])

    # .4.1 基线实验
    add_label_para(doc, "4.1 基线实验（Baseline）　", topic["baseline_desc"])
    add_table(
        doc,
        ["基线编号", "基线内容", "设置要点", "作用与定位"],
        topic["baselines"],
        widths=[0.78, 1.95, 2.25, 1.82],
        font_size=8.4,
    )

    # .4.2 改进对比实验
    add_label_para(doc, "4.2 改进对比实验（Proposed vs. Baselines/SOTA）　", topic["improve_desc"])
    add_table(
        doc,
        ["对比编号", "对比问题", "改进方法与对照", "数据与条件", "输出与判定"],
        topic["improve_experiments"],
        widths=[0.62, 1.10, 1.52, 1.72, 1.84],
        font_size=8.2,
    )

    # .4.3 消融实验
    add_label_para(doc, "4.3 消融实验（Ablation）　", topic["ablation_desc"])
    add_table(
        doc,
        ["消融编号", "被消融对象", "消融方式", "观测指标", "预期结论"],
        topic["ablations"],
        widths=[0.62, 1.28, 1.55, 1.48, 1.87],
        font_size=8.2,
    )

    # .4.4 实验间对比关系与依赖
    add_label_para(doc, "4.4 实验间对比关系与依赖　", topic["relation_desc"])
    add_table(
        doc,
        ["实验组", "依赖的实验", "对照关系", "在证据链中的角色"],
        topic["relations"],
        widths=[0.95, 1.45, 2.30, 2.10],
        font_size=8.4,
    )

    # .4.5 关键实验逐项执行细则
    add_label_para(doc, "4.5 关键实验逐项执行细则　", topic["detail_desc"])
    add_table(
        doc,
        ["实验", "变量与水平", "数据划分与规模", "训练 / 评测配置", "判定标准与产出图表"],
        topic["exp_details"],
        widths=[0.72, 1.55, 1.45, 1.68, 1.50],
        font_size=8.0,
    )

    # .4.6 指标口径与统计检验
    add_label_para(doc, "4.6 指标口径与统计检验方法　", topic["stat_desc"])
    add_table(
        doc,
        ["指标类别", "具体指标", "计算口径", "统计方法"],
        topic["stat_specs"],
        widths=[1.05, 1.85, 2.35, 1.65],
        font_size=8.3,
    )

    # .4.7 计算与实验量估计
    add_label_para(doc, "4.7 实验量与算力预算　", topic["budget_desc"])
    add_table(
        doc,
        ["实验组", "训练次数估计", "单次成本", "累计预算", "优先级"],
        topic["budgets"],
        widths=[1.35, 1.35, 1.55, 1.35, 1.30],
        font_size=8.4,
    )

    # .4.8 已有结果现状
    add_label_para(doc, "4.8 已有实验结果现状　", "")
    add_callout(doc, "现场实测（远端 GPU 环境）：", topic["current_results"], fill=topic.get("result_fill", SOFT_RED))
    add_body(doc, topic["current_interpretation"])

    # ---------- .5 数据集 指标与统计规范 ----------
    add_heading(doc, f"{idx}.5 数据集 指标与统计规范", 2)
    add_table(doc, ["类别", "具体安排"], topic["data_metrics"], widths=[1.30, 5.60], font_size=9)

    # ---------- .6 论文大纲 ----------
    add_heading(doc, f"{idx}.6 论文大纲", 2)
    add_numbered(doc, topic["outline"])

    # ---------- .7 预期成果与验收闸门 ----------
    add_heading(doc, f"{idx}.7 预期成果与验收闸门", 2)
    add_table(doc, ["成果", "预期交付", "进入论文结论的最低证据"], topic["outputs"],
              widths=[1.20, 2.85, 2.85], font_size=8.8)

    # ---------- .8 风险 失败判据与降级路线 ----------
    add_heading(doc, f"{idx}.8 风险 失败判据与降级路线", 2)
    add_table(doc, ["主要风险", "失败判据", "处理或降级方案"], topic["risks"],
              widths=[1.72, 2.18, 3.00], font_size=8.8)
    add_label_para(doc, "关键难点　", topic["difficulties"])
    add_label_para(doc, "适用边界　", topic["applicable_boundary"])

    # ---------- .9 计划安排 ----------
    add_heading(doc, f"{idx}.9 计划安排", 2)
    add_table(doc, ["阶段", "工作内容", "阶段交付", "通过闸门"], topic["schedule"],
              widths=[0.90, 2.95, 1.65, 1.40], font_size=8.6)

    # ---------- .10 文献与代码锚点 ----------
    add_heading(doc, f"{idx}.10 文献与代码锚点", 2)
    add_label_para(doc, "文献证据锚点　", "")
    add_bullets(doc, topic["lit_anchors"])
    add_label_para(doc, "代码与工程锚点　", "")
    add_bullets(doc, topic["code_anchors"])
