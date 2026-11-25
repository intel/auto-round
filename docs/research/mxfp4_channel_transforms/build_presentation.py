# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Build an offline, editable research deck; verify its evidence and packaging."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import statistics
import tempfile
import unicodedata
import zipfile
from copy import deepcopy
from pathlib import Path
from xml.etree import ElementTree

import pymupdf
from PIL import Image
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parent
WIDTH, HEIGHT = 16.0, 9.0
FONT = "Noto Sans CJK SC"
ENGLISH_FONT = "Liberation Sans"
CJK = re.compile(r"[\u3400-\u9fff]")
COLORS = {
    "navy": "15263F",
    "ink": "1A2C44",
    "teal": "008784",
    "blue": "3569D6",
    "orange": "BA651D",
    "muted": "576B81",
    "pale": "F0F5FA",
    "line": "DCE5EF",
    "white": "FFFFFF",
    "mint": "E7F5F1",
    "warm": "FFF1E3",
}


def rgb(name: str) -> RGBColor:
    return RGBColor.from_string(COLORS.get(name, name))


def text(
    slide,
    value: str,
    x: float,
    y: float,
    w: float,
    h: float,
    size: float = 20,
    color: str = "ink",
    bold: bool = False,
    font: str = FONT,
    align=PP_ALIGN.LEFT,
):
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = shape.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.margin_left = frame.margin_right = Inches(0.02)
    frame.margin_top = frame.margin_bottom = Inches(0.02)
    frame.vertical_anchor = MSO_ANCHOR.TOP
    for index, line in enumerate(value.split("\n")):
        paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        paragraph.text = line
        paragraph.alignment = align
        paragraph.space_after = Pt(7)
        paragraph.line_spacing = 1.15
        paragraph.font.name = font
        paragraph.font.size = Pt(size)
        paragraph.font.bold = bold
        paragraph.font.color.rgb = rgb(color)
    return shape


def panel(slide, x, y, w, h, fill="pale", line=None, rounded=True):
    kind = MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE
    shape = slide.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
    if rounded:
        shape.adjustments[0] = 0.12
    shape.fill.solid()
    shape.fill.fore_color.rgb = rgb(fill)
    if line:
        shape.line.color.rgb = rgb(line)
    else:
        shape.line.fill.background()
    return shape


def card(slide, title, body, x, y, w, h, accent="teal", size=20):
    panel(slide, x, y, w, h)
    panel(slide, x, y, 0.06, h, fill=accent, rounded=False)
    text(slide, title, x + 0.23, y + 0.18, w - 0.46, 0.55, 24, accent, True)
    text(slide, body, x + 0.23, y + 0.92, w - 0.46, h - 1.05, size)


def arrow(slide, x, y, w=0.4, h=0.35, color="teal"):
    shape = slide.shapes.add_shape(MSO_SHAPE.CHEVRON, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = rgb(color)
    shape.line.fill.background()
    return shape


def table(slide, headers, rows, x, y, widths, height, size=17):
    shape = slide.shapes.add_table(
        len(rows) + 1,
        len(headers),
        Inches(x),
        Inches(y),
        Inches(sum(widths)),
        Inches(height),
    )
    grid = shape.table
    for index, width in enumerate(widths):
        grid.columns[index].width = Inches(width)
    for row_index, values in enumerate([headers, *rows]):
        for col_index, value in enumerate(values):
            cell = grid.cell(row_index, col_index)
            cell.text = str(value)
            cell.margin_left = cell.margin_right = Inches(0.12)
            cell.margin_top = cell.margin_bottom = Inches(0.08)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            cell.fill.fore_color.rgb = rgb("navy" if row_index == 0 else ("pale" if row_index % 2 else "white"))
            for paragraph in cell.text_frame.paragraphs:
                paragraph.font.name = FONT
                paragraph.font.size = Pt(size if row_index else size - 1)
                paragraph.font.bold = row_index == 0
                paragraph.font.color.rgb = rgb("white" if row_index == 0 else "ink")
                paragraph.space_after = Pt(3)
    return shape


def picture(slide, asset, x, y, w, h, description):
    path = ROOT / asset
    with Image.open(path) as image:
        ratio = image.width / image.height
    draw_w, draw_h = min(w, h * ratio), min(h, w / ratio)
    shape = slide.shapes.add_picture(
        str(path),
        Inches(x + (w - draw_w) / 2),
        Inches(y + (h - draw_h) / 2),
        width=Inches(draw_w),
        height=Inches(draw_h),
    )
    shape.name = asset
    shape._element.nvPicPr.cNvPr.set("descr", description)
    return shape


def bar_chart(slide, categories, series, x, y, w, h, maximum=100, minimum=0, suffix=""):
    data = CategoryChartData()
    data.categories = categories
    for name, values in series:
        data.add_series(name, values)
    chart = slide.shapes.add_chart(
        XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(x), Inches(y), Inches(w), Inches(h), data
    ).chart
    chart.has_legend = len(series) > 1
    if chart.has_legend:
        chart.legend.position = XL_LEGEND_POSITION.BOTTOM
        chart.legend.font.name = FONT
        chart.legend.font.size = Pt(14)
    chart.chart_style = 10
    chart.font.name = FONT
    chart.font.size = Pt(16)
    chart.value_axis.minimum_scale = minimum
    chart.value_axis.maximum_scale = maximum
    chart.value_axis.has_major_gridlines = True
    chart.value_axis.tick_labels.font.name = FONT
    chart.value_axis.tick_labels.font.size = Pt(14)
    chart.category_axis.tick_labels.font.name = FONT
    chart.category_axis.tick_labels.font.size = Pt(15)
    plot = chart.plots[0]
    plot.gap_width = 70
    plot.has_data_labels = True
    plot.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
    plot.data_labels.number_format = f'0.00"{suffix}"' if suffix else "0.00"
    plot.data_labels.font.name = FONT
    plot.data_labels.font.size = Pt(16)
    for index, values in enumerate(chart.series):
        values.format.fill.solid()
        values.format.fill.fore_color.rgb = rgb(("teal", "blue", "orange")[index % 3])
        values.format.line.fill.background()
    return chart


class Deck:
    def __init__(self, language="CN"):
        self.language = language
        self.presentation = Presentation()
        self.presentation.slide_width = Inches(WIDTH)
        self.presentation.slide_height = Inches(HEIGHT)
        self.evidence = json.loads((ROOT / "evidence.json").read_text())
        self.sources = {item["id"]: item for item in self.evidence["sources"]}
        self.manifest = []
        properties = self.presentation.core_properties
        properties.title = "AutoRound：通道重排、混合 MXFP 激活与 HMT"
        properties.subject = "Intel XPU research and implementation design; no new-feature measurements"
        properties.author = "AutoRound research/design"
        properties.keywords = "MXFP4, MXFP8, HMT, Intel XPU, GEMM, grouped GEMM, research"

    def slide(self, title, subtitle="", section="设计方案", refs=(), note="", dark=False):
        slide = self.presentation.slides.add_slide(self.presentation.slide_layouts[6])
        slide.background.fill.solid()
        slide.background.fill.fore_color.rgb = rgb("navy" if dark else "white")
        page = len(self.presentation.slides)
        foreground = "white" if dark else "ink"
        text(slide, f"AUTO ROUND  /  {section}", 0.7, 0.32, 14.6, 0.35, 12, "teal", True)
        text(slide, title, 0.7, 0.88, 14.6, 0.75, 34, foreground, True)
        if subtitle:
            text(slide, subtitle, 0.7, 1.7, 14.6, 0.6, 16, "line" if dark else "muted")
        panel(slide, 0.7, 8.33, 14.6, 0.012, "muted" if dark else "line", rounded=False)
        text(
            slide,
            "研究与开发设计 | Intel XPU | 2026-11-05 | 非新功能实测",
            0.7,
            8.48,
            12.8,
            0.3,
            11,
            "line" if dark else "muted",
        )
        text(slide, f"{page:02d}", 14.45, 8.43, 0.8, 0.35, 13, foreground, align=PP_ALIGN.RIGHT)
        source_notes = []
        for source_id in refs:
            source = self.sources[source_id]
            source_notes.append(
                f"{source_id}: {source['title']}\n"
                f"Paper: {source['paper_url'] or 'Documentation'}\n"
                f"Repository: {source['repository']} @ {source['revision']}\n"
                + "\n".join(f"{path}: {item['url']}" for path, item in source["files"].items())
            )
        slide.notes_slide.notes_text_frame.text = (
            f"{title}\n\n{note}\n\nEvidence boundary: pinned author implementations, author-released "
            "figures/data and local source inspected. Direct arXiv full-text access failed. "
            "No new-feature model or XPU benchmark was run.\n\n" + "\n\n".join(source_notes)
        )
        self.manifest.append({"page": page, "title": title, "section": section, "source_ids": list(refs)})
        return slide

    def localize_to_english(self):
        translations = json.loads((ROOT / "slides_EN_content.json").read_text())

        def translate(value):
            if not CJK.search(value):
                return value
            if value not in translations:
                raise ValueError(f"Missing English translation: {value!r}")
            result = translations[value]
            if CJK.search(result):
                raise ValueError(f"Non-English translation: {value!r}")
            return result

        for slide in self.presentation.slides:
            for shape in slide.shapes:
                frames = [shape.text_frame] if shape.has_text_frame else []
                if shape.has_table:
                    frames.extend(cell.text_frame for row in shape.table.rows for cell in row.cells)
                for frame in frames:
                    translated = translate(frame.text)
                    style = deepcopy(frame.paragraphs[0]._p.pPr)
                    frame.clear()
                    for index, line in enumerate(translated.split("\n")):
                        paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
                        paragraph.text = line
                        if paragraph._p.pPr is not None:
                            paragraph._p.remove(paragraph._p.pPr)
                        if style is not None:
                            paragraph._p.insert(0, deepcopy(style))
                        paragraph.font.name = ENGLISH_FONT
                if shape.has_chart:
                    chart = shape.chart
                    data = CategoryChartData()
                    data.categories = [translate(category.label) for category in chart.plots[0].categories]
                    for series in chart.series:
                        data.add_series(translate(series.name), series.values)
                    chart.replace_data(data)
                    chart.font.name = ENGLISH_FONT
                    chart.value_axis.tick_labels.font.name = ENGLISH_FONT
                    chart.category_axis.tick_labels.font.name = ENGLISH_FONT
                    chart.plots[0].data_labels.font.name = ENGLISH_FONT
            for paragraph in slide.notes_slide.notes_text_frame.paragraphs:
                paragraph.text = translate(paragraph.text)
                paragraph.font.name = ENGLISH_FONT
        for entry in self.manifest:
            entry["title"] = translate(entry["title"])
            entry["section"] = translate(entry["section"])
        self.presentation.core_properties.title = translate(self.presentation.core_properties.title)
        self.presentation.core_properties.language = "en-US"

    def save(self):
        if self.language == "EN":
            self.localize_to_english()
        license_text = "\n\n".join(path.read_text() for path in sorted((ROOT / "third_party").glob("*.txt")))
        last_notes = self.presentation.slides[-1].notes_slide.notes_text_frame
        last_notes.text += "\n\nCOMPLETE THIRD-PARTY NOTICES AND LICENSES\n\n" + license_text
        self.presentation.save(ROOT / f"slides_{self.language}.pptx")
        manifest_path(self.language).write_text(json.dumps(self.manifest, ensure_ascii=False, indent=2) + "\n")


def manifest_path(language):
    return ROOT / ("slides_manifest.json" if language == "CN" else "slides_manifest_EN.json")


def build(language="CN"):
    deck = Deck(language)
    data = deck.evidence["data"]

    slide = deck.slide(
        "通道重排 × 混合 MXFP 激活 × HMT",
        "AutoRound 的算法与工程设计：以 Intel GPU / XPU 为主",
        "研究设计",
        dark=True,
        note="用户已确认 HMT 为 Hadamard Transform，主要硬件为 Intel GPU/XPU。本交付只做研究和设计。",
    )
    text(slide, "让 4 位量化更准\n也让实现路径可验证", 0.75, 2.8, 13.8, 2.0, 44, "white", True)
    text(
        slide,
        "F1  固定 K 轴重排 + MXFP4\nF2  主体 A4 + 少量 A8\nHMT  可选、成对补偿、联合优化",
        0.8,
        5.3,
        12.8,
        1.8,
        24,
        "line",
    )
    text(slide, "13 个固定来源 · 作者图表 · 真实代码落点 · 分阶段开发方案", 0.8, 7.5, 14, 0.45, 17, "teal")

    slide = deck.slide(
        "建议：一份计划，两个功能，三层落地", "先验证数值与精度，再选择真实硬件路径。默认不升级权重位宽。"
    )
    card(
        slide,
        "F1 / 改变分组",
        "离线学习固定置换\n激活与权重列同步重排\n重新形成 32 元素块\n先用完整 W4A4",
        0.7,
        2.65,
        4.65,
        3.9,
    )
    card(
        slide,
        "F2 / 分配精度",
        "一次 gather 得到两区\n主体激活 MXFP4\n少量激活 MXFP8\n默认两区权重均 MXFP4",
        5.67,
        2.65,
        4.65,
        3.9,
        "blue",
    )
    card(
        slide,
        "HMT / 可选增强",
        "分区后先旋转低区\n固定 H32 作为起点\n两侧严格配对补偿\n学习旋转放后续",
        10.64,
        2.65,
        4.65,
        3.9,
        "orange",
    )
    text(slide, "参考 QDQ  →  XPU 解码融合  →  经能力探测的原生运算", 1.0, 7.1, 14, 0.65, 26, "teal", True)

    slide = deck.slide(
        "先讲证据边界：哪些知道，哪些尚未证明", "作者结果、理论推导、开发目标不能混为一谈。", section="证据与先行工作"
    )
    table(
        slide,
        ["证据级别", "本次已核验", "不能据此声称"],
        [
            ["代码事实", "AutoRound 固定提交；作者算法/内核实现", "相关路径已完成生产验证"],
            ["作者报告", "原始论文图、作者 CSV、固定 README 结果", "新功能已在 XPU 达到同等收益"],
            ["推导与设计", "配对变换公式、预算和对齐计算", "实际延迟与理论字节成本相同"],
            ["待验证", "候选算法、门槛、开发估算", "已取得精度提升或加速"],
        ],
        0.7,
        2.6,
        [2.4, 6.1, 6.1],
        3.8,
        19,
    )
    text(
        slide,
        "访问限制：论文站点直连失败；以作者公开实现、作者论文图与原始数据核验。\n完整提交与哈希见 evidence.json；不声称通读全部全文或穷尽所有工作。",
        0.9,
        6.95,
        14.0,
        1.05,
        19,
        "muted",
    )

    slide = deck.slide(
        "最接近的先行工作，决定创新应放在哪里",
        "不是提出新格式；重点是 AutoRound 优化、XPU 和 MoE 的联合落地。",
        section="证据与先行工作",
        refs=("R02", "R03", "R04", "R05", "R08", "R09", "R10", "R11", "R12"),
    )
    table(
        slide,
        ["方向", "最相关工作", "本方案的增量"],
        [
            ["通道重排", "RPTQ；DuQuant", "重构损失驱动、局部性约束"],
            ["MXFP4 + 旋转", "DuQuant++；FP-Quant", "与 AutoRound rounding 联合选型"],
            ["少量高精度特征", "LLM.int8 / QUIK / Atom", "默认 W4，激活 A4/A8"],
            ["混合 MX 通道", "MicroMix（ICLR 2026 分支）", "XPU 双分区 ABI、MoE 共享/分桶"],
            ["数据感知变换", "SpinQuant；WUSH", "成本匹配的进阶探索，不纳入首期"],
        ],
        0.7,
        2.6,
        [3.0, 5.3, 6.3],
        4.85,
        19,
    )

    slide = deck.slide(
        "QuaRot：成对旋转，而不是单边处理异常值",
        "作者原图 / INT4 工作；用于说明旋转思想，不是 MXFP4 或 XPU 新结果。",
        section="论文图",
        refs=("R05",),
        note="Unmodified author-released img/fig1.png, Apache-2.0. Caption/source license retained.",
    )
    picture(slide, "assets/quarot_fig1.png", 0.8, 2.55, 14.4, 4.7, "QuaRot author-released figure 1")
    text(slide, "核心迁移原则：函数保持不变，量化误差才是优化对象。", 0.9, 7.5, 14, 0.5, 23, "teal", True)
    text(slide, "来源 R05 · spcl/QuaRot@5008669 · Apache-2.0", 0.9, 8.0, 14, 0.25, 11, "muted")

    slide = deck.slide(
        "DuQuant：旋转 → 置换 → 再旋转",
        "作者原图：先处理块内异常值，再平衡块间分布。不能据此认定随机 shuffle 一定有效。",
        section="论文图",
        refs=("R03", "R04"),
        note="Unmodified DuQuant imgs/duquant.png under MIT. This is the original integer work, not DuQuant++ MXFP4 measurements.",
    )
    picture(
        slide, "assets/duquant_method.png", 0.7, 2.45, 9.7, 5.65, "DuQuant rotation-permutation-rotation method figure"
    )
    card(
        slide,
        "两种分组目标",
        "无 HMT：相近幅度聚集\n有 HMT：测试异常值均摊\n二者都用完整 QDQ 验证",
        10.7,
        2.6,
        4.6,
        2.65,
        size=18,
    )
    card(
        slide,
        "更直接的 MXFP4 对照",
        "DuQuant++（2026）\n块长 32、细粒度旋转\n额外尺度约定需重新对齐",
        10.7,
        5.55,
        4.6,
        2.45,
        "blue",
        18,
    )
    text(slide, "来源 R03 · MIT；R04 为独立的后续 MXFP4 实现。", 0.9, 8.03, 9.8, 0.24, 11, "muted")

    slide = deck.slide(
        "DuQuant 的原始数据表：保留格式和上下文",
        "W4A4 整数实验；1-* / 2-* 为 LLaMA-1 / LLaMA-2。PPL 越低越好。",
        section="论文数据",
        refs=("R03",),
        note="Author-released imgs/ppl.png, MIT. NaNs and missing entries remain unchanged. No cross-paper fairness is inferred.",
    )
    picture(slide, "assets/duquant_ppl.png", 0.7, 2.55, 11.4, 5.6, "Original DuQuant W4A4 perplexity table")
    card(
        slide,
        "怎样读这张表",
        "比较同一模型和数据集\n保留 NaN 与缺失值\n不把 INT4 写成 MXFP4\n不外推 Intel 加速比",
        12.4,
        2.8,
        2.9,
        4.8,
        "orange",
        17,
    )
    text(slide, "来源 R03 · Hsu1023/DuQuant@d56cfc6 · MIT", 0.9, 8.02, 11.2, 0.25, 11, "muted")

    slide = deck.slide(
        "MicroMix：与 F2 最接近的直接先行实现",
        "原创概念重绘，不复制该仓库图片/源码；固定核验 micromix 分支。",
        section="现有实现",
        refs=("R10",),
        note="Public source inspected, not redistributed. utilize.py sorts channels and allocates aligned ratios; w4a8.cu declares MXFP8 A and MXFP4 B on SM120.",
    )
    for x, label, body, fill in [
        (0.8, "通道统计", "静态收集激活\n生成固定排序", "teal"),
        (5.9, "比例与重排", "按精度分成连续区\n原实现含 4 / 6 / 8", "blue"),
        (11.0, "混合 GEMM", "W4A4 / W4A8 等\nBlackwell 内核", "orange"),
    ]:
        panel(slide, x, 3.0, 4.15, 2.3, "pale")
        text(slide, label, x + 0.2, 3.2, 3.75, 0.6, 25, fill, True)
        text(slide, body, x + 0.2, 4.0, 3.75, 1.0, 20)
    arrow(slide, 5.1, 3.95)
    arrow(slide, 10.2, 3.95)
    text(
        slide, "核验到真实 MXFP8 激活 × MXFP4 权重；不应把高分支默认改成 W8A8。", 0.9, 5.95, 14.2, 0.8, 23, "teal", True
    )
    text(
        slide,
        "原实现按 128 通道对齐比例；XPU 对齐需重新确定。\nMVP 只做 4/8 两区，避免同时引入 FP6 的数值与部署复杂度。",
        0.9,
        7.0,
        14.2,
        0.9,
        20,
        "muted",
    )

    slide = deck.slide(
        "直接的 MXFP4 证据：固定 HMT 仍有改进空间",
        "WUSH 作者数据 · Llama-3.1-8B-Instruct · 13 个 Platinum 任务宏平均准确率（%）",
        section="论文数据",
        refs=("R12",),
        note="Recomputed from author CSV: per-record 13-task mean, then method mean. RTN-WUSH 79.34116; H 76.41440; GPTQ-WUSH 79.25155; BF16 85.33235. No independent-seed inference.",
    )
    results = data["wush_platinum"]["results"]
    bar_chart(
        slide,
        ["BF16", "RTN + H", "RTN + WUSH", "GPTQ + WUSH"],
        [("准确率", [row["macro_mean_percent"] for row in results])],
        0.7,
        2.6,
        9.3,
        4.95,
    )
    table(
        slide,
        ["作者方法", "准确率", "记录"],
        [
            [label, f"{result['macro_mean_percent']:.2f}", result["records"]]
            for label, result in zip(["BF16", "RTN-H", "RTN-WUSH", "GPTQ-WUSH"], results, strict=True)
        ],
        10.4,
        2.8,
        [2.4, 1.5, 1.0],
        2.7,
        16,
    )
    card(
        slide,
        "+2.93 个百分点",
        "RTN-WUSH 相对 RTN-H\n仍落后 BF16 5.99 点\n记录数不等于独立种子",
        10.4,
        5.8,
        4.9,
        2.25,
        size=17,
    )
    text(slide, "来源 R12 · 原始 CSV 随包提供 · 图表为可编辑 PowerPoint chart", 0.9, 8.01, 14, 0.24, 11, "muted")

    slide = deck.slide(
        "格式不混淆：MXFP4、MXFP8、NVFP4", "每个数字比较都要先对齐数值 profile。", refs=("R01", "R04", "R11")
    )
    table(
        slide,
        ["格式", "元素", "共享尺度", "本方案"],
        [
            ["MXFP4", "E2M1；最大幅值 6", "E8M0，32 元素", "低区与默认全部权重"],
            ["MXFP8", "默认 E4M3；E5M2 可选", "E8M0，32 元素", "高激活分区"],
            ["NVFP4", "E2M1", "不同块长与尺度约定", "独立对照，不混报结果"],
        ],
        0.7,
        2.6,
        [2.2, 4.5, 4.2, 3.7],
        3.05,
        19,
    )
    text(slide, "MXFP4 幅值：0 · 0.5 · 1 · 1.5 · 2 · 3 · 4 · 6", 0.9, 6.15, 14.2, 0.6, 26, "teal", True)
    text(
        slide,
        "ARK / core floor / rceil / CUDA HMT 并非完全同约定。\n冻结舍入、零块、符号零、尺度、nibble 顺序；禁止校准与导出悄悄切换 profile。",
        0.9,
        7.1,
        14.1,
        1.0,
        20,
    )

    slide = deck.slide("不变量：激活和权重列必须成对变换", "约定 X[M,K]、W[N,K]，Y = X W^T + b。")
    panel(slide, 0.8, 2.7, 14.4, 1.35, "mint")
    text(slide, "X' = X[:, p]      W' = W[:, p]      X' (W')^T = X W^T", 1.1, 3.1, 13.8, 0.7, 26, "teal", True)
    card(slide, "正交 HMT", "R R^T = I\nX' = X P R\nW' = W P R", 0.8, 4.55, 6.85, 2.8, size=23)
    card(slide, "一般可逆 T", "X' = X T\nW' = W T⁻ᵀ\nWUSH 不能套用两侧同乘 T", 8.0, 4.55, 7.2, 2.8, "blue", 23)
    text(slide, "这是 K 轴重排，不是 token 路由；最终 N 维输出不需要逆置换。", 1.0, 7.65, 14, 0.45, 21, "orange", True)

    slide = deck.slide(
        "F1：要优化跨块成员，而非“打乱”本身", "没有跨量化块变化时，shuffle 应作为负对照，而不是收益来源。"
    )
    labels = [
        ("0.4", "teal"),
        ("7.8", "orange"),
        ("0.3", "teal"),
        ("6.2", "orange"),
        ("0.5", "teal"),
        ("9.1", "orange"),
    ]
    text(slide, "原顺序（示意）", 0.8, 2.65, 3.0, 0.5, 22, bold=True)
    for index, (label, color) in enumerate(labels):
        panel(slide, 3.8 + index * 1.65, 2.6, 1.45, 0.8, "warm" if color == "orange" else "mint")
        text(slide, label, 3.92 + index * 1.65, 2.77, 1.2, 0.4, 22, color, True)
    text(slide, "候选重排", 0.8, 3.8, 3.0, 0.5, 22, bold=True)
    for index, (label, color) in enumerate(sorted(labels, key=lambda pair: float(pair[0]))):
        panel(slide, 3.8 + index * 1.65, 3.75, 1.45, 0.8, "warm" if color == "orange" else "mint")
        text(slide, label, 3.92 + index * 1.65, 3.92, 1.2, 0.4, 22, color, True)
    card(
        slide,
        "无 HMT：测试幅度相近的分组",
        "按 RMS / 分位数 / log 幅度排序\n权重列范数与量化残差共同打分",
        0.8,
        5.2,
        6.9,
        2.5,
        size=20,
    )
    card(
        slide,
        "有 HMT：还要测试异常值均摊",
        "zigzag / 块交换 / 局部性约束\n同样用完整 W4A4 输出损失验证",
        8.0,
        5.2,
        7.2,
        2.5,
        "blue",
        20,
    )
    text(slide, "数值与小方块仅为原创示意；实际共享尺度块为 32 元素。", 0.9, 8.03, 14.0, 0.23, 11, "muted")

    slide = deck.slide("F1 搜索：用重构损失，而不只用 amax", "先廉价筛选，再短程 AutoRound；排列在调优期间固定。")
    table(
        slide,
        ["候选", "复杂度/目标", "筛选原则"],
        [
            ["identity / random", "必须有的基线", "固定种子；跨块与块内对照"],
            ["RMS / 分位数排序", "O(K log K)", "幅度相近；加入权重敏感度"],
            ["zigzag + HMT", "异常值跨旋转块均摊", "与聚类路线直接对比"],
            ["局部交换", "限制 128/256/512 窗口", "精度收益 + 实际 gather 成本"],
        ],
        0.7,
        2.6,
        [4.1, 4.1, 6.4],
        3.65,
        19,
    )
    text(slide, "损失 = ||原层输出 − 完整 QDQ 层输出||² / (||原层输出||² + ε)", 0.9, 6.7, 14.1, 0.65, 25, "teal", True)
    text(slide, "QKV / gate-up 等共享输入的消费者联合归一化打分，避免重复 gather。", 0.9, 7.65, 14.0, 0.45, 20)

    slide = deck.slide(
        "grouped GEMM：M 轴 routing 与 K 轴 gather 分开",
        "两种重排不要混为一种；保留 top-k 权重和最终 scatter 对应关系。",
    )
    stages = [
        ("token × top-k", "原 token id\nrouter 权重"),
        ("按 expert 排序", "M 轴 gather\n生成专家区间"),
        ("通道计划", "K 轴 gather\n共享/专家计划"),
        ("grouped GEMM", "双分区累加\nscatter/reduce"),
    ]
    for index, (title, body) in enumerate(stages):
        x = 0.8 + index * 3.85
        panel(slide, x, 3.15, 3.15, 2.5, "pale")
        text(slide, title, x + 0.16, 3.38, 2.83, 0.65, 22, "teal", True)
        text(slide, body, x + 0.16, 4.25, 2.83, 1.0, 20)
        if index < 3:
            arrow(slide, x + 3.28, 4.2)
    text(
        slide,
        "MVP：专家组共享计划；gate/up 共用预处理。\n下一步：每专家 mask + 固定 K8 档位，按形状/profile/HMT 分桶。",
        0.9,
        6.45,
        14.2,
        1.2,
        23,
    )
    text(slide, "down-proj 是每专家自己的中间激活，不能无条件复用 gate/up 计划。", 0.9, 7.85, 14.1, 0.35, 18, "orange")

    slide = deck.slide(
        "F2：一遍 gather，两段连续输入，沿 K 求和", "默认是低区 W4A4 + 高区 W4A8；不是两个完整 K 的 GEMM。"
    )
    panel(slide, 0.8, 3.9, 2.5, 1.0, "navy")
    text(slide, "X[M,K]", 1.1, 4.16, 1.9, 0.5, 27, "white", True)
    arrow(slide, 3.55, 4.2)
    panel(slide, 4.25, 2.85, 4.0, 1.35, "mint")
    text(slide, "X[:,S4] → Q4", 4.5, 3.25, 3.5, 0.55, 26, "teal", True)
    panel(slide, 4.25, 5.0, 4.0, 1.35, "warm")
    text(slide, "X[:,S8] → Q8", 4.5, 5.4, 3.5, 0.55, 26, "orange", True)
    arrow(slide, 8.5, 3.35)
    arrow(slide, 8.5, 5.5)
    panel(slide, 9.15, 2.85, 3.0, 1.35, "pale")
    text(slide, "× Q4(W[:,S4])ᵀ", 9.35, 3.23, 2.6, 0.7, 20)
    panel(slide, 9.15, 5.0, 3.0, 1.35, "pale")
    text(slide, "× Q4(W[:,S8])ᵀ", 9.35, 5.38, 2.6, 0.7, 20)
    arrow(slide, 12.38, 4.2)
    panel(slide, 13.0, 3.9, 2.2, 1.15, "navy")
    text(slide, "求和 + b", 13.2, 4.18, 1.8, 0.55, 24, "white", True)
    text(slide, "S4 ∩ S8 = ∅，S4 ∪ S8 = 全部 K；bias 只加一次。", 0.95, 7.3, 14.0, 0.7, 25, "teal", True)

    slide = deck.slide(
        "高分支有三种语义，必须分开报告", "权重存储格式、实际操作数格式、量化精度不是同一个概念。", refs=("R10",)
    )
    table(
        slide,
        ["方案", "高区权重", "解释"],
        [
            ["默认 W4A8", "原始权重 → MXFP4", "只提高激活精度；严格对应需求"],
            ["W4 码展开为 W8 操作数", "E2M1 → E4M3 无损码转换", "权重数值不变；运行时缓存可能增大"],
            ["可选 W8A8", "原始权重 → MXFP8", "权重量化也升级；额外预算、单独消融"],
        ],
        0.7,
        2.65,
        [3.6, 4.7, 6.3],
        3.25,
        20,
    )
    text(
        slide, "E2M1 的有限值可被 E4M3 精确表示，但尺度布局与设备数值仍需验证。", 0.9, 6.45, 14.2, 0.7, 22, "teal", True
    )
    text(
        slide,
        "若不支持 A8W4：显式选择无损展开或解码路径。\n不允许把 INT4/INT8 内核改名后当作 MXFP4/MXFP8。",
        0.9,
        7.25,
        14.1,
        0.9,
        20,
    )

    slide = deck.slide(
        "给谁 A8：量化误差增益优先，幅度只是代理", "移动通道会改变整个共享尺度块，通道收益并不天然可加。"
    )
    card(slide, "廉价候选分数", "RMS × 权重列范数\n高分位 / 尾部频次\nA4→A8 的误差改善", 0.8, 2.75, 4.65, 3.0, size=21)
    card(
        slide,
        "必须真实复算",
        "重新 gather 与分组\n两区全部尺度重算\n计算完整层输出误差",
        5.65,
        2.75,
        4.65,
        3.0,
        "blue",
        21,
    )
    card(
        slide, "再做预算选择", "边界通道交换\n固定实际 K8 比较\n全模型收益/成本分配", 10.5, 2.75, 4.7, 3.0, "orange", 21
    )
    text(slide, "s[j] ≈ ||Q4(W)[:,j]||² × E[err4(xj)² − err8(xj)²]", 1.0, 6.45, 14.0, 0.7, 23, "teal", True)
    text(slide, "仅是固定尺度/分组下的筛选代理；最终选择由完整分区 QDQ 决定。", 1.0, 7.55, 14.0, 0.55, 21, "muted")

    slide = deck.slide(
        "小比例不是连续旋钮：对齐决定真实预算",
        "理论 activation payload，未计 padding、索引、布局、展开缓存；不是性能实测。",
    )
    categories = ["0%", "3.125%", "6.25%", "12.5%", "25%"]
    ratios = [0, 0.03125, 0.0625, 0.125, 0.25]
    bar_chart(
        slide, categories, [("bit / 元素", [4.25 + 4 * ratio for ratio in ratios])], 0.7, 2.65, 7.0, 4.9, maximum=8
    )
    table(
        slide,
        ["K", "目标", "K8", "实际"],
        [
            ["768", "1%", "32", "4.17%"],
            ["2048", "1%", "32", "1.56%"],
            ["4096", "1%", "64", "1.56%"],
            ["4096", "5%", "224", "5.47%"],
        ],
        8.15,
        2.8,
        [1.65, 1.65, 1.65, 2.0],
        3.5,
        19,
    )
    text(slide, "4.25 + 4ρ bit / activation 元素", 8.35, 6.65, 6.5, 0.6, 24, "teal", True)
    text(slide, "6.25% A8 → 4.50 bit，payload +5.88%。\n默认不增加 checkpoint 权重位宽。", 8.35, 7.35, 6.5, 0.8, 19)

    slide = deck.slide("HMT 组合：先明确 mask 属于哪个坐标基", "推荐从 H1 开始；PR 与 RP 一般不相等。")
    table(
        slide,
        ["方案", "顺序", "风险/作用"],
        [
            ["H0", "分区 → A4/A8，无 HMT", "分离两个功能自身收益"],
            ["H1 / 推荐", "原坐标选高区 → 分区 → 仅低区 HMT", "高敏感通道不被混回低区"],
            ["H2", "全输入 HMT → 重新校准选通道", "旧坐标 mask 无效"],
            ["H3", "分区 → 两区独立 HMT", "高区小 K 的 padding/开销"],
            ["H4", "旋转 → zigzag → 再旋转", "DuQuant 风格；两次变换不免费"],
        ],
        0.7,
        2.6,
        [2.6, 6.3, 5.7],
        4.85,
        18,
    )
    text(
        slide,
        "先比较 H32 / 随机符号 H32，再扩至 128/256/512；MX 量化 group 始终 32。",
        0.9,
        7.7,
        14.2,
        0.45,
        19,
        "teal",
        True,
    )

    slide = deck.slide(
        "统一搜索：离散计划与 AutoRound 交替优化", "恢复/重建调优状态，避免旧排列的梯度、尺度与统计被误用。"
    )
    steps = [
        ("01", "统计与候选", "路由加权统计\nidentity / 聚类 / zigzag"),
        ("02", "分区与 HMT", "在正确坐标选 S8\n重算尺度与完整 QDQ"),
        ("03", "短程调优", "前 2–4 名跑 AutoRound\n独立选型集重排"),
        ("04", "成本选择", "实测 XPU 成本 Pareto\n冻结计划、最终调优"),
    ]
    for index, (number, title, body) in enumerate(steps):
        x = 0.8 + index * 3.85
        text(slide, number, x, 2.7, 3.2, 0.8, 38, "teal", True)
        text(slide, title, x, 3.7, 3.25, 0.7, 25, bold=True)
        text(slide, body, x, 4.65, 3.25, 1.5, 20)
        if index < 3:
            arrow(slide, x + 3.25, 4.15)
    panel(slide, 0.8, 6.75, 14.4, 1.0, "mint")
    text(slide, "J = 重构损失 + λ × 完整实测延迟 + μ × 驻留字节", 1.05, 7.0, 13.9, 0.65, 26, "teal", True)

    slide = deck.slide(
        "进阶方向：从固定 HMT 到数据感知变换",
        "WUSH 作者原图：不是所有最优候选都是正交旋转。",
        section="论文图 / 探索",
        refs=("R12", "R06"),
        note="Author figure plots/1_toy_example.pdf rasterized without modifying labels/data, Apache-2.0. Theoretical assumptions do not establish this task's optimality.",
    )
    picture(slide, "assets/wush_toy.png", 0.8, 2.5, 5.6, 5.55, "WUSH paper figure 1 author visualization")
    card(
        slide,
        "为什么值得比较",
        "同时利用权重与激活二阶统计\n固定 Hadamard 不一定最优\n但理论假设不等于任务保证",
        7.0,
        2.7,
        8.2,
        2.5,
        size=21,
    )
    card(
        slide,
        "为什么不作为首期默认",
        "非正交变换需 W T⁻ᵀ 补偿\n稠密小矩阵与尺度成本要实测\n与可学习旋转按真实成本对齐",
        7.0,
        5.5,
        8.2,
        2.5,
        "blue",
        21,
    )
    text(slide, "来源 R12 · Apache-2.0 · 左图可放大阅读，解释为本报告原创。", 0.9, 8.03, 14.1, 0.22, 11, "muted")

    slide = deck.slide(
        "论文速度图能说明什么，不能说明什么",
        "WUSH 作者图 2：M=1024，逐层相对 BF16；不是端到端或 Intel 数据。",
        section="论文图 / 证据边界",
        refs=("R12",),
        note="Author plots/2_speedup.pdf and plotting arrays inspected. Device model is not established by this figure/script; do not infer it from supported-backend lists.",
    )
    picture(
        slide,
        "assets/wush_speedup.png",
        0.8,
        2.55,
        14.4,
        4.45,
        "WUSH paper figure 2 per-layer speedups vs BF16 at M=1024",
    )
    text(slide, "变换与量化开销必须计入，但仍可能保留部分内核收益。", 0.95, 7.25, 14.0, 0.5, 23, "teal", True)
    text(
        slide,
        "图/绘图脚本未明确 GPU 型号，不据此做跨硬件比较。来源 R12 · Apache-2.0。",
        0.95,
        7.95,
        14,
        0.25,
        12,
        "muted",
    )

    slide = deck.slide(
        "AutoRound 落点：复用已有机制，不另起一套", "基线 5db1082；新的计划是前处理/量化包装层，不是新 scalar dtype。"
    )
    table(
        slide,
        ["层", "现有机制", "拟开发增量"],
        [
            ["方案与数值", "schemes.py；data_type/mxfp.py", "显式计划 schema；复用 4/8 QDQ"],
            ["组合与校准", "composer；BasePreprocessor", "计划冻结、参考失效、坐标统计"],
            ["调优与 grouped", "wrapper；grouped_experts.py", "分区 QDQ、梯度、计划签名/分桶"],
            ["XPU 内核", "ARK HMT + MXFP4 原语", "gather / sign / 双区输出 + GEMM 消费"],
            ["保存与推理", "qlinear_fp；formats；backend", "两区 packed buffers、版本化重载"],
        ],
        0.7,
        2.6,
        [3.0, 5.3, 6.3],
        4.8,
        18,
    )
    text(slide, "开关默认关闭；旧模型、旧 preset、旧导出结果保持不变。", 0.9, 7.75, 14.1, 0.4, 22, "teal", True)

    slide = deck.slide("三个真实缺口：hook、packed 消费者、后端能力", "代码存在，不等于已经自动进入模型推理路径。")
    card(
        slide,
        "grouped 的 hook 限制",
        "现有在线变换 hook 会退出\n部分 grouped 快速路径。\n需显式计划阶段，而非只挂 hook。",
        0.8,
        2.8,
        4.65,
        3.8,
        size=20,
    )
    card(
        slide,
        "已有 HMT 原语",
        "XPU 原语产出 packed FP4\n但不证明已接入 dense / MoE\nMXFP4 activation GEMM。",
        5.65,
        2.8,
        4.65,
        3.8,
        "blue",
        20,
    )
    card(
        slide,
        "现有 W4A8 并非 MX",
        "ARK 路径是 INT4 存储\n加 INT8 DPAS 计算；\n部分 FP8 代码待硬件验证。",
        10.5,
        2.8,
        4.7,
        3.8,
        "orange",
        20,
    )
    text(
        slide,
        "把实际分派路径、fallback 原因、workspace 与代码版本纳入 benchmark 记录。",
        0.9,
        7.25,
        14.2,
        0.8,
        23,
        "teal",
        True,
    )

    slide = deck.slide(
        "Intel XPU：按代际核验原生计算，而不是按名称猜测",
        "oneDNN 固定主分支提交 4149450；不是用户已安装版本的保证。",
        section="XPU 实现",
        refs=("R13",),
        note="Datatype table marks native compute vs conversion. Native datatype support does not guarantee mixed A8W4, E8M0 scale layouts, or ragged grouped GEMM.",
    )
    hardware_rows = [
        [row["uarch"], "原生" if row["f8"] == "native" else "转换", "原生" if row["f4_e2m1"] == "native" else "转换"]
        for row in data["onednn_hardware"]["rows"]
    ]
    table(slide, ["架构", "FP8", "FP4 E2M1"], hardware_rows, 0.7, 2.6, [6.2, 4.2, 4.2], 4.35, 21)
    text(slide, "还必须实际创建 primitive：A8W4？块尺度布局？grouped / ragged？", 0.9, 7.2, 14.2, 0.6, 24, "teal", True)
    text(
        slide,
        "记录 GPU id / 驱动 / oneAPI / oneDNN / ARK 版本；unimplemented 显式失败。",
        0.9,
        7.95,
        14.0,
        0.25,
        13,
        "muted",
    )

    slide = deck.slide(
        "内核路线：参考、解码融合、能力通过后的原生",
        "同一 packed ABI 与同一数值契约，三种执行层次。",
        section="XPU 实现",
    )
    table(
        slide,
        ["层次", "实现", "收益/限制"],
        [
            ["Reference", "FP32 变换 + QDQ；拼接后高精度 GEMM", "定义正确性；不能用其推理时间代表低位内核"],
            [
                "转换计算设备",
                "load 解码 FP4/FP8；组尺度；FP16/BF16 DPAS",
                "减少存储和完整解码 workspace；不宣称原生 FP4 算力",
            ],
            ["原生设备", "支持的 block-scaled 算子；双分支或融合累加", "每个 operand/shape/profile 都需能力验证"],
        ],
        0.7,
        2.6,
        [2.9, 6.0, 5.7],
        3.9,
        19,
    )
    panel(slide, 0.8, 7.0, 14.4, 0.95, "mint")
    text(slide, "融合目标：routing + gather + sign + HMT + quantize + pack", 1.0, 7.25, 14.0, 0.6, 24, "teal", True)

    slide = deck.slide(
        "持久化：一份版本化计划贯穿校准、调优、推理",
        "拟议 schema，不是当前可调用 API；不把未知字段塞入会过滤字段的 from_dict。",
    )
    table(
        slide,
        ["元数据", "张量 buffers", "加载校验"],
        [
            ["version / profile / transform_order", "实际 permutation、sign", "维度、排列完整性、哈希"],
            ["mask_basis / logical K / padded K4,K8", "两区 packed 权重 + E8M0 尺度", "分区不重叠、覆盖 K、组边界"],
            ["共享计划关系 / 已变换权重标记", "可选无损展开缓存不当作原权重", "应用一次；不重复旋转"],
        ],
        0.7,
        2.6,
        [5.0, 5.0, 4.6],
        3.35,
        18,
    )
    text(slide, "save → shard/offload → load → generate 都必须保留同一计划。", 0.9, 6.5, 14.1, 0.6, 25, "teal", True)
    text(
        slide,
        "safetensors buffers；旧 checkpoint 兼容；TP 首期仅 rank-local K 置换。\n不支持的 backend 明确报错，只有显式允许时才做带日志的 reference 回退。",
        0.9,
        7.25,
        14.1,
        0.85,
        19,
    )

    slide = deck.slide(
        "正确性：先查不变量，再查编码，最后查网络",
        "不同累加顺序不强求字节相同；匹配 profile 的 packed 编码应严格一致。",
    )
    table(
        slide,
        ["层级", "关键测试"],
        [
            ["数学", "配对置换、正交/逆转置、只变一侧负测试、归一化因子"],
            ["编码", "tie、全零/负零、极值/次正规、NaN/Inf、E8M0 保留值"],
            ["分区", "空/全 S8、尾块、重复/越界索引拒绝、bias 一次、padding 保留"],
            ["grouped", "空专家、热点、top-k scatter、共享/独立计划、签名/fallback"],
            ["生命周期", "梯度/STE、quantize-save-load、即时 shard、meta/offload、旧模型"],
        ],
        0.7,
        2.6,
        [2.5, 12.1],
        4.8,
        19,
    )
    text(
        slide,
        "禁用开关时行为不变；不把随机模型的小规模通过当作真实模型精度结论。",
        0.9,
        7.8,
        14.0,
        0.4,
        19,
        "teal",
        True,
    )

    slide = deck.slide(
        "实验：公平预算，独立数据，真实 MoE 负载", "tiny → dense → MoE；覆盖 prefill 与 decode，而不只测大 M GEMM。"
    )
    table(
        slide,
        ["维度", "推荐矩阵"],
        [
            ["基线", "BF16；原 MXFP4 RTN / AutoRound；F1、F2、组合，各自 ±HMT"],
            ["预算", "相同实际 K8、全部 W4、同校准量/调优轮数；W8 高区单列"],
            ["形状", "M=1/8/32/128/1024；K=768/2048/4096/11008 等真实投影"],
            ["MoE", "E=8/64/128，top-k=2/8；均衡、长尾、空专家、热点"],
            ["泛化", "校准/选型/最终评测分离；代码数学、长上下文、分布外；≥3 种子"],
        ],
        0.7,
        2.6,
        [2.5, 12.1],
        4.8,
        18,
    )
    text(slide, "记录路由覆盖；强制覆盖样本不能悄悄替代真实 token 权重。", 0.9, 7.8, 14, 0.4, 21, "teal", True)

    slide = deck.slide(
        "建议门槛：是研发目标，不是已经取得的结果", "明确区分质量研究成功与硬件算术加速成立。", section="验收设计"
    )
    card(
        slide,
        "质量门槛 / 建议",
        "2 dense + 1 MoE\n选型集中位层误差下降 ≥20%\n最终任务/PPL 无实质回退\n至少一项实际改善",
        0.8,
        2.8,
        7.0,
        3.6,
        size=21,
    )
    card(
        slide,
        "工程门槛 / 建议",
        "先测试 K8≤6.25% 档\n完整投影延迟 ≤1.10× 基线\n记录峰值/驻留显存\n整模型 Pareto 决策",
        8.1,
        2.8,
        7.1,
        3.6,
        "blue",
        21,
    )
    text(
        slide,
        "总延迟 = routing + gather + HMT + 量化 + 两分支 GEMM + 求和 + scatter",
        0.9,
        6.95,
        14.1,
        0.7,
        22,
        "teal",
        True,
    )
    text(
        slide,
        "同机、同预算、预热后中位数/p95；kernel / block / TTFT / TPOT 分开报告。",
        0.9,
        7.85,
        14.1,
        0.3,
        17,
        "muted",
    )

    slide = deck.slide(
        "探索优先级：先解决四个可证伪的问题",
        "后续的学习旋转、动态 mask 和 MXFP6，都应受真实成本约束。",
        section="探索路线",
    )
    table(
        slide,
        ["优先级", "方向", "可证伪问题"],
        [
            ["P0", "误差增益选 A8", "相同 K8 下，是否胜过只按最大幅度？"],
            ["P0", "HMT + A8", "互补还是替代？能否降低必要 A8 预算？"],
            ["P0", "局部性约束置换", "能否保留精度，同时减少随机 gather 损失？"],
            ["P0", "MoE 共享/专家计划", "独立计划收益是否覆盖 top-k 重复处理？"],
            ["P1", "指数边界、全模型预算、WUSH/SpinQuant", "同成本下是否更好？正确补偿是否可部署？"],
            ["P2", "动态 mask、W8/低秩旁路、MXFP6", "额外精度与额外成本能否单独归因？"],
        ],
        0.7,
        2.55,
        [1.5, 5.5, 7.6],
        5.25,
        18,
    )

    slide = deck.slide(
        "开发里程碑：数值、dense、grouped、持久化闭环",
        "人日是工程估算，不是排期承诺；算法/内核工程师可部分并行。",
        section="开发计划",
    )
    table(
        slide,
        ["阶段", "交付物", "依赖", "估算"],
        [
            ["M0", "设备能力 / profile / packed ABI / baseline", "实际 XPU", "2–3 人日"],
            ["M1", "CPU reference / F1-F2 dense / 负对照", "M0", "4–6"],
            ["M2", "HMT / 搜索 / AutoRound wrapper", "M1", "5–8"],
            ["M3", "XPU 前处理融合 / dense 双分支", "M0/M1；HMT 需 M2", "6–10"],
            ["M4", "专家共享计划 / 分桶 / decode", "M2/M3", "6–10"],
            ["M5", "导出重载 / shard-TP / 完整评测", "M2/M3；MoE 需 M4", "5–8"],
        ],
        0.7,
        2.55,
        [1.5, 7.0, 3.6, 2.5],
        5.25,
        17,
    )

    slide = deck.slide(
        "最终决策：用 Pareto 证据，而不是单项宣传",
        "不把“更准”“更小”“更快”当作同一个结论。",
        section="推荐决策",
        dark=True,
    )
    text(slide, "先做 M0 / M1，再决定值得投入哪条内核路线。", 0.85, 2.9, 14.3, 0.85, 31, "white", True)
    text(
        slide,
        "保留：同预算下真实精度改善的静态计划。\n推进：完整成本受控、能导出重载的 XPU 路径。\n停用：校准集好看但泛化变差、或 gather 成本失控的候选。",
        0.9,
        4.25,
        14.0,
        2.2,
        25,
        "line",
    )
    text(slide, "核心贡献：AutoRound × 通道预算 × HMT × Intel XPU / MoE", 0.9, 7.2, 14.1, 0.6, 27, "teal", True)

    short_names = [
        "Microscaling / OCP MX",
        "RPTQ",
        "DuQuant",
        "DuQuant++",
        "QuaRot",
        "SpinQuant",
        "LLM.int8() / bitsandbytes",
        "QUIK",
        "Atom",
        "MicroMix",
        "FP-Quant / MR-GPTQ",
        "WUSH",
        "oneDNN datatype / MXFP8 tutorial",
    ]
    sources = deck.evidence["sources"]
    for page, (start, end) in enumerate([(0, 7), (7, 13)], start=1):
        subset = sources[start:end]
        slide = deck.slide(
            f"参考来源 {page}/2：论文入口与固定实现版本",
            "完整作者标题、URL、文件 SHA256、许可和证据说明见 evidence.json 及逐页备注。",
            section="参考与许可",
            refs=tuple(item["id"] for item in subset),
        )
        rows = [
            [
                source["id"],
                short_names[start + index],
                source["paper_url"].rsplit("/", 1)[-1] if source["paper_url"] else "主分支文档",
                source["revision"][:7],
            ]
            for index, source in enumerate(subset)
        ]
        table(slide, ["ID", "工作/资料", "论文编号", "核验提交"], rows, 0.7, 2.55, [1.0, 7.2, 3.6, 2.8], 4.9, 18)
        text(
            slide,
            "图像仅复用明确 MIT / Apache 作者资产；MicroMix / Atom / SpinQuant 不复制插图。",
            0.9,
            7.85,
            14,
            0.3,
            15,
            "muted",
        )

    slide = deck.slide(
        "附录：WUSH 作者原始任务明细表",
        "Llama-3.1-8B-Instruct · 图 10 作者发布版本 · 原数据/标签未改动 · 可放大阅读",
        section="附录 / 原始论文表",
        refs=("R12",),
        note="Apache-2.0 author figure, rasterized from plots/3_platinum_bench/10_llama-3.1-8b_table.pdf. Full third-party licenses follow in these notes.",
    )
    picture(
        slide,
        "assets/wush_llama_table.png",
        0.7,
        2.4,
        14.6,
        5.6,
        "WUSH author figure 10 complete Platinum task accuracy table",
    )
    text(slide, "来源 R12 · Apache-2.0 · 完整许可在本页备注；PDF 包含许可附件。", 0.9, 8.02, 14, 0.24, 11, "muted")
    deck.save()
    print(f"Built {len(deck.presentation.slides)} editable slides: {ROOT / f'slides_{language}.pptx'}")


def finalize_pdf(language="CN"):
    path = ROOT / f"slides_{language}.pdf"
    with tempfile.TemporaryDirectory(prefix="slides-pdf-", dir=ROOT) as directory:
        output = Path(directory) / "finalized.pdf"
        with pymupdf.open(path) as document:
            for attachment in sorted((ROOT / "third_party").glob("*.txt")):
                if attachment.name in document.embfile_names():
                    document.embfile_del(attachment.name)
                document.embfile_add(
                    attachment.name,
                    attachment.read_bytes(),
                    filename=attachment.name,
                    ufilename=attachment.name,
                    desc="Complete third-party license or source attribution for reproduced figures/data.",
                )
            document.save(output, garbage=4, deflate=True)
        output.replace(path)
    print("Embedded complete third-party licenses and notices in the PDF.")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized_text(value):
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", value))


def verify_evidence(evidence):
    for item in evidence["figures"]:
        path = ROOT / item["asset"]
        if sha256(path) != item["asset_sha256"]:
            raise ValueError(f"Figure differs from evidence manifest: {path.name}")
    source = evidence["data"]["wush_platinum"]
    if sha256(ROOT / source["asset"]) != source["sha256"]:
        raise ValueError("Author CSV differs from evidence manifest")
    with (ROOT / source["asset"]).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    for result in source["results"]:
        method = f"{source['model']}-BF16" if result["method"] == "BF16" else result["method"]
        selected = [row for row in rows if row["model"] == source["model"] and row["pretty_name"] == method]
        values = [statistics.mean(float(row[name]) for name in source["task_columns"]) for row in selected]
        if len(values) != result["records"] or not math.isclose(
            statistics.mean(values), result["macro_mean_percent"], abs_tol=1e-10
        ):
            raise ValueError(f"CSV aggregation mismatch: {method}")


def verify_documents():
    documents = [(ROOT / filename).read_text() for filename in ("summary.md", "summary_CN.md")]
    sections = [re.findall(r"^(#{2,3}) (\d+(?:\.\d+)?)(?:\.\s|\s)", item, re.MULTILINE) for item in documents]
    if sections[0] != sections[1]:
        raise ValueError("English/Chinese section structure differs")
    definitions = [dict(re.findall(r"^\[(R\d+)\]: (.+)$", item, re.MULTILINE)) for item in documents]
    if definitions[0] != definitions[1] or len(definitions[0]) != 13:
        raise ValueError("English/Chinese source definitions differ")
    for document in documents:
        for target in re.findall(r"\]\(([^)]+)\)", document):
            if target.startswith(("https://", "http://", "#")):
                continue
            path = ROOT / target.split("#", 1)[0]
            if not path.exists():
                raise ValueError(f"Missing local document link: {target}")
    if re.findall(r"```json\n(.*?)```", documents[0], re.DOTALL) != re.findall(
        r"```json\n(.*?)```", documents[1], re.DOTALL
    ):
        raise ValueError("English/Chinese proposed checkpoint metadata differs")


def check(language="CN"):
    evidence = json.loads((ROOT / "evidence.json").read_text())
    verify_evidence(evidence)
    verify_documents()
    presentation_path = ROOT / f"slides_{language}.pptx"
    presentation = Presentation(presentation_path)
    manifest = json.loads(manifest_path(language).read_text())
    if len(presentation.slides) != len(manifest) or len(manifest) != 36:
        raise ValueError("Unexpected slide count")
    chart_count, picture_count, table_count = 0, 0, 0
    for index, slide in enumerate(presentation.slides):
        if manifest[index]["title"] not in "\n".join(shape.text for shape in slide.shapes if shape.has_text_frame):
            raise ValueError(f"Missing title on slide {index + 1}")
        if "Evidence boundary" not in slide.notes_slide.notes_text_frame.text:
            raise ValueError(f"Missing evidence notes on slide {index + 1}")
        for shape in slide.shapes:
            if min(shape.left, shape.top) < 0:
                raise ValueError(f"Negative shape position on slide {index + 1}")
            if shape.left + shape.width > presentation.slide_width + 10:
                raise ValueError(f"Shape exceeds right edge on slide {index + 1}")
            if shape.top + shape.height > presentation.slide_height + 10:
                raise ValueError(f"Shape exceeds bottom edge on slide {index + 1}")
            chart_count += bool(shape.has_chart)
            table_count += bool(shape.has_table)
            picture_count += shape.shape_type == 13
    with zipfile.ZipFile(presentation_path) as archive:
        if any("vbaproject" in name.lower() for name in archive.namelist()):
            raise ValueError("Unexpected macro content")
        for name in archive.namelist():
            if not name.endswith(".rels"):
                continue
            for relation in ElementTree.fromstring(archive.read(name)):
                if relation.attrib.get("TargetMode") == "External" and relation.attrib.get("Type", "").endswith(
                    "/image"
                ):
                    raise ValueError("Unexpected externally loaded image")
        image_hashes = {
            hashlib.sha256(archive.read(name)).hexdigest()
            for name in archive.namelist()
            if name.startswith("ppt/media/")
        }
        if not {figure["asset_sha256"] for figure in evidence["figures"]}.issubset(image_hashes):
            raise ValueError("An author figure is missing or changed in the deck")
        if language == "EN":
            for name in archive.namelist():
                if name.endswith(".xml") and name.startswith(
                    ("ppt/slides/", "ppt/notesSlides/", "ppt/charts/", "docProps/")
                ):
                    content = "".join(ElementTree.fromstring(archive.read(name)).itertext())
                    if CJK.search(content):
                        raise ValueError(f"Chinese text remains in English presentation: {name}")
                if name.startswith("ppt/embeddings/") and name.endswith(".xlsx"):
                    with zipfile.ZipFile(io.BytesIO(archive.read(name))) as workbook:
                        for item in workbook.namelist():
                            if item.endswith(".xml"):
                                content = "".join(ElementTree.fromstring(workbook.read(item)).itertext())
                                if CJK.search(content):
                                    raise ValueError(f"Chinese chart data remains in English presentation: {name}")
    notes = presentation.slides[-1].notes_slide.notes_text_frame.text
    for path in (ROOT / "third_party").glob("*.txt"):
        if path.read_text() not in notes:
            raise ValueError(f"Complete license missing from PowerPoint notes: {path.name}")
    if chart_count < 2 or table_count < 10 or picture_count < 6:
        raise ValueError("Missing editable charts/tables or author figures")
    chart = next(shape.chart for shape in presentation.slides[8].shapes if shape.has_chart)
    expected = [row["macro_mean_percent"] for row in evidence["data"]["wush_platinum"]["results"]]
    if len(chart.series[0].values) != len(expected) or not all(
        math.isclose(actual, value, abs_tol=1e-10)
        for actual, value in zip(chart.series[0].values, expected, strict=True)
    ):
        raise ValueError("Editable WUSH chart differs from author CSV")
    with pymupdf.open(ROOT / f"slides_{language}.pdf") as document:
        if len(document) != len(manifest):
            raise ValueError("PDF/PPT slide count differs")
        for index, page in enumerate(document):
            content = page.get_text()
            if language == "EN" and CJK.search(content):
                raise ValueError(f"Chinese text remains in English PDF on slide {index + 1}")
            if len(content.strip()) < 50 or normalized_text(manifest[index]["title"]) not in normalized_text(content):
                raise ValueError(f"Missing/unreadable PDF text on slide {index + 1}")
            for shape in presentation.slides[index].shapes:
                values = [shape.text] if shape.has_text_frame else []
                if shape.has_table:
                    values.extend(cell.text for row in shape.table.rows for cell in row.cells)
                for value in values:
                    if normalized_text(value) not in normalized_text(content):
                        raise ValueError(f"Missing PDF text on slide {index + 1}: {value[:80]!r}")
        if not {"Apache-2.0.txt", "DuQuant_MIT.txt", "THIRD_PARTY_NOTICES.txt"}.issubset(document.embfile_names()):
            raise ValueError("PDF license attachments missing")
        for path in (ROOT / "third_party").glob("*.txt"):
            if document.embfile_get(path.name) != path.read_bytes():
                raise ValueError(f"PDF license attachment differs: {path.name}")
    print(
        f"Verified: {len(manifest)} slides, {chart_count} editable charts, {table_count} editable tables, "
        f"{picture_count} author figures; asset hashes, CSV, paired docs, local links, PDF text and licenses."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--language", type=str.upper, choices=("CN", "EN"), default="CN", help="Output language (default: CN)"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Check all persistent deliverables, including rendered PDF")
    mode.add_argument("--finalize-pdf", action="store_true", help="Attach licenses after LibreOffice conversion")
    args = parser.parse_args()
    if args.check:
        check(args.language)
    elif args.finalize_pdf:
        finalize_pdf(args.language)
    else:
        build(args.language)


if __name__ == "__main__":
    main()
