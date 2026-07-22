#!/usr/bin/env python3

from __future__ import annotations

import csv
import html
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SWEEP_DIR = ROOT / "bench_sparse_topk_sweeps"
OUT_SVG = SWEEP_DIR / "sparse_e2e_vs_sagev1_hnd_nhd.svg"

LAYOUTS = ("HND", "NHD")
SEQ_LENS = (16000, 32000, 75600)
TOPKS = (0.5, 0.4, 0.3, 0.2, 0.1)
TOPK_COLORS = {
    0.5: "#355C7D",
    0.4: "#4E7A57",
    0.3: "#C06C84",
    0.2: "#F67280",
    0.1: "#F8B55F",
}


def load_results() -> dict[str, dict[int, dict[float, float]]]:
    results: dict[str, dict[int, dict[float, float]]] = {layout: {} for layout in LAYOUTS}
    for layout in LAYOUTS:
        for seq_len in SEQ_LENS:
            csv_path = SWEEP_DIR / f"bench_sparse_topk_{layout}_seqlen{seq_len}_qtile256.csv"
            with csv_path.open() as f:
                rows = list(csv.DictReader(f))
            seq_results: dict[float, float] = {}
            for row in rows:
                if row["mode"] != "sparse_qtile256_row64k_e2e" or row["status"] != "ok":
                    continue
                topk = float(row["requested_topk"])
                seq_results[topk] = float(row["speedup_vs_sagev1"])
            results[layout][seq_len] = seq_results
    return results


def make_svg(results: dict[str, dict[int, dict[float, float]]]) -> str:
    width = 1400
    height = 760
    margin_left = 90
    margin_right = 30
    margin_top = 80
    margin_bottom = 90
    panel_gap = 40
    panel_width = (width - margin_left - margin_right - panel_gap) / 2
    panel_height = height - margin_top - margin_bottom
    max_val = max(
        results[layout][seq_len][topk]
        for layout in LAYOUTS
        for seq_len in SEQ_LENS
        for topk in TOPKS
    )
    y_max = max(4.0, round(max_val + 0.5, 1))
    plot_bg = "#FCFBF7"
    axis_color = "#2B2B2B"
    grid_color = "#D8D2C8"
    text_color = "#1F1F1F"

    def y_to_px(value: float) -> float:
        return margin_top + panel_height - (value / y_max) * panel_height

    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
    )
    parts.append(f'<rect width="{width}" height="{height}" fill="{plot_bg}"/>')
    parts.append(
        f'<text x="{width/2}" y="38" text-anchor="middle" font-size="26" font-family="sans-serif" fill="{text_color}">'
        "Sparse E2E Speedup vs dense_sagev1"
        "</text>"
    )
    parts.append(
        f'<text x="{width/2}" y="62" text-anchor="middle" font-size="14" font-family="sans-serif" fill="{text_color}">'
        "Baseline = dense_sagev1 for each (layout, seq_len); bars show sparse_qtile256_row64k_e2e speedup"
        "</text>"
    )

    legend_x = width - 360
    legend_y = 26
    for idx, topk in enumerate(TOPKS):
        x = legend_x + idx * 68
        parts.append(f'<rect x="{x}" y="{legend_y}" width="16" height="16" fill="{TOPK_COLORS[topk]}"/>')
        parts.append(
            f'<text x="{x+22}" y="{legend_y+13}" font-size="13" font-family="sans-serif" fill="{text_color}">'
            f"topk={topk:.1f}</text>"
        )

    for panel_idx, layout in enumerate(LAYOUTS):
        panel_x = margin_left + panel_idx * (panel_width + panel_gap)
        panel_y = margin_top
        parts.append(
            f'<text x="{panel_x + panel_width/2}" y="{panel_y - 18}" text-anchor="middle" font-size="20" '
            f'font-family="sans-serif" fill="{text_color}">{layout}</text>'
        )
        parts.append(
            f'<rect x="{panel_x}" y="{panel_y}" width="{panel_width}" height="{panel_height}" '
            f'fill="white" stroke="#C9C2B8" stroke-width="1"/>'
        )

        for tick in range(0, int(y_max) + 1):
            y = y_to_px(tick)
            parts.append(
                f'<line x1="{panel_x}" y1="{y:.1f}" x2="{panel_x + panel_width}" y2="{y:.1f}" '
                f'stroke="{grid_color}" stroke-width="1"/>'
            )
            parts.append(
                f'<text x="{panel_x - 10}" y="{y + 5:.1f}" text-anchor="end" font-size="12" '
                f'font-family="sans-serif" fill="{text_color}">{tick}x</text>'
            )

        baseline_y = y_to_px(1.0)
        parts.append(
            f'<line x1="{panel_x}" y1="{baseline_y:.1f}" x2="{panel_x + panel_width}" y2="{baseline_y:.1f}" '
            f'stroke="#AA3A3A" stroke-width="2" stroke-dasharray="6 4"/>'
        )
        parts.append(
            f'<text x="{panel_x + panel_width - 6}" y="{baseline_y - 6:.1f}" text-anchor="end" font-size="12" '
            f'font-family="sans-serif" fill="#AA3A3A">sagev1 baseline</text>'
        )

        group_width = panel_width / len(SEQ_LENS)
        bar_cluster_width = group_width * 0.66
        bar_width = bar_cluster_width / len(TOPKS)

        for seq_idx, seq_len in enumerate(SEQ_LENS):
            group_x0 = panel_x + seq_idx * group_width
            cluster_x0 = group_x0 + (group_width - bar_cluster_width) / 2
            center_x = group_x0 + group_width / 2
            parts.append(
                f'<text x="{center_x}" y="{panel_y + panel_height + 28}" text-anchor="middle" font-size="13" '
                f'font-family="sans-serif" fill="{text_color}">{seq_len}</text>'
            )
            for topk_idx, topk in enumerate(TOPKS):
                value = results[layout][seq_len][topk]
                x = cluster_x0 + topk_idx * bar_width + 2
                y = y_to_px(value)
                h = panel_y + panel_height - y
                parts.append(
                    f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width - 4:.1f}" height="{h:.1f}" '
                    f'fill="{TOPK_COLORS[topk]}"/>'
                )
                parts.append(
                    f'<text x="{x + (bar_width - 4)/2:.1f}" y="{y - 6:.1f}" text-anchor="middle" font-size="10" '
                    f'font-family="sans-serif" fill="{text_color}">{value:.2f}x</text>'
                )

        parts.append(
            f'<text x="{panel_x + panel_width/2}" y="{height - 24}" text-anchor="middle" font-size="14" '
            f'font-family="sans-serif" fill="{text_color}">Sequence Length</text>'
        )

    parts.append(
        f'<text transform="translate(24 {margin_top + panel_height/2}) rotate(-90)" text-anchor="middle" '
        f'font-size="14" font-family="sans-serif" fill="{text_color}">Speedup vs dense_sagev1</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def main() -> None:
    results = load_results()
    OUT_SVG.write_text(make_svg(results))
    print(html.escape(str(OUT_SVG)))


if __name__ == "__main__":
    main()
