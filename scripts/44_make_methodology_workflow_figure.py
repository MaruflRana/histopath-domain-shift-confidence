"""Render the Milestone 9G methodology workflow figure deterministically.

This script uses author-defined, locked protocol facts only. It imports no project,
dataset, model, checkpoint, prediction, training, calibration, or evaluation module.
It creates vector SVG/PDF files and a high-resolution PNG review copy without
photographic, synthetic, or generative-image content.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "results" / "figures"
STEM = "figure_methodology_workflow"
FIGURE_TITLE = "Locked multi-hospital development and reserved-center evaluation workflow"


COLORS = {
    "ink": "#202124",
    "muted": "#5F6368",
    "line": "#4A4A4A",
    "source_fill": "#DCEAF7",
    "source_edge": "#0072B2",
    "development_fill": "#E3F2E9",
    "development_edge": "#009E73",
    "reserved_fill": "#FDE9D9",
    "reserved_edge": "#D55E00",
    "neutral_fill": "#F2F2F2",
    "neutral_edge": "#666666",
    "model_fill": "#EEE7F3",
    "model_edge": "#8E5A9E",
    "freeze_fill": "#FFF4CC",
    "freeze_edge": "#E69F00",
    "analysis_fill": "#EAF2F8",
    "analysis_edge": "#33658A",
    "white": "#FFFFFF",
}


def add_box(
    ax,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    body: str = "",
    *,
    fill: str,
    edge: str,
    title_size: float = 7.0,
    body_size: float = 5.8,
    dashed: bool = False,
    linewidth: float = 1.15,
    title_y_pad: float = 0.18,
    body_y_pad: float = 0.48,
    radius: float = 0.08,
    body_family: str = "DejaVu Sans",
):
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.025,rounding_size={radius}",
        facecolor=fill,
        edgecolor=edge,
        linewidth=linewidth,
        linestyle=(0, (4, 2)) if dashed else "solid",
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        x + 0.12,
        y + height - title_y_pad,
        title,
        ha="left",
        va="top",
        fontsize=title_size,
        fontweight="bold",
        color=COLORS["ink"],
        zorder=3,
    )
    if body:
        ax.text(
            x + 0.12,
            y + height - body_y_pad,
            body,
            ha="left",
            va="top",
            fontsize=body_size,
            fontfamily=body_family,
            color=COLORS["ink"],
            linespacing=1.16,
            zorder=3,
        )
    return patch


def add_arrow(ax, start, end, *, connectionstyle="arc3", color=None, linewidth=1.35):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=9,
        linewidth=linewidth,
        color=color or COLORS["line"],
        connectionstyle=connectionstyle,
        shrinkA=2,
        shrinkB=2,
        zorder=4,
    )
    ax.add_patch(arrow)


def add_stage_label(ax, x, y, number, label):
    ax.text(
        x,
        y,
        f"{number}  {label}",
        ha="left",
        va="center",
        fontsize=7.4,
        fontweight="bold",
        color=COLORS["ink"],
        zorder=5,
    )


def render() -> dict[str, str | int | float]:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 6.0,
            "axes.linewidth": 0.0,
            "svg.hashsalt": "milestone-9g-methodology-workflow-v1",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(7.5, 5.4), dpi=600)
    fig.patch.set_facecolor(COLORS["white"])
    ax.set_facecolor(COLORS["white"])
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 10.8)
    ax.axis("off")

    ax.text(
        7.5,
        10.52,
        FIGURE_TITLE,
        ha="center",
        va="center",
        fontsize=10.2,
        fontweight="bold",
        color=COLORS["ink"],
    )
    ax.text(
        7.5,
        10.22,
        "Prespecified development sequence, freeze gate, and one-shot final analysis",
        ha="center",
        va="center",
        fontsize=6.2,
        color=COLORS["muted"],
    )

    # Stage 1: dataset roles and the locked final-center boundary.
    add_box(
        ax,
        0.25,
        7.35,
        4.8,
        2.55,
        "1  Dataset and hospital roles",
        fill=COLORS["neutral_fill"],
        edge=COLORS["neutral_edge"],
        title_size=6.5,
        body_y_pad=0.42,
    )
    add_box(
        ax,
        0.42,
        7.62,
        1.30,
        1.58,
        "Source / ID",
        "Centers 0, 3, 4\nTraining\nID validation",
        fill=COLORS["source_fill"],
        edge=COLORS["source_edge"],
        title_size=5.6,
        body_size=4.45,
        body_y_pad=0.43,
    )
    add_box(
        ax,
        1.80,
        7.62,
        1.34,
        1.58,
        "Development\nOOD",
        "Center 1\nModel\ncomparison\nTemperature\nfitting\nDevelopment-only\nevidence",
        fill=COLORS["development_fill"],
        edge=COLORS["development_edge"],
        title_size=4.9,
        body_size=3.35,
        body_y_pad=0.60,
    )
    add_box(
        ax,
        3.22,
        7.62,
        1.63,
        1.58,
        "Reserved final",
        "Center 2\nOne authorized\nfinal evaluation\nNo development use",
        fill=COLORS["reserved_fill"],
        edge=COLORS["reserved_edge"],
        title_size=5.25,
        body_size=4.0,
        dashed=True,
        linewidth=1.55,
        body_y_pad=0.43,
    )
    ax.text(
        4.04,
        7.48,
        "LOCKED UNTIL FINAL EVALUATION",
        ha="center",
        va="center",
        fontsize=4.65,
        fontweight="bold",
        color=COLORS["reserved_edge"],
    )

    # Stage 2: transparent description of the capped source construction.
    add_box(
        ax,
        5.18,
        7.35,
        3.02,
        2.55,
        "2  Center-stratified\nsource construction",
        "Sample by center x label\n\nTraining cache: 300 per cell\n1,800 total\nValidation cache: 75 per cell\n450 total\n\nPrevents center-label\nconfounding",
        fill=COLORS["neutral_fill"],
        edge=COLORS["neutral_edge"],
        title_size=5.85,
        body_size=4.45,
        body_y_pad=0.72,
    )

    # Stage 3: two matched branches and the shared controls.
    add_box(
        ax,
        8.50,
        7.35,
        6.25,
        2.55,
        "3  Matched model development",
        fill=COLORS["model_fill"],
        edge=COLORS["model_edge"],
        title_size=6.5,
        body_y_pad=0.42,
    )
    add_box(
        ax,
        8.68,
        8.35,
        2.64,
        0.95,
        "Branch A | Matched\ncontrol",
        "Center-stratified ERM",
        fill=COLORS["white"],
        edge=COLORS["source_edge"],
        title_size=4.55,
        body_size=4.55,
        body_y_pad=0.60,
    )
    add_box(
        ax,
        11.49,
        8.35,
        3.08,
        0.95,
        "Branch B | Predeclared primary\ncandidate",
        "GroupDRO-by-center",
        fill=COLORS["white"],
        edge=COLORS["model_edge"],
        title_size=4.45,
        body_size=4.55,
        body_y_pad=0.62,
    )
    add_box(
        ax,
        8.68,
        7.53,
        5.89,
        0.67,
        "Shared controls",
        "ResNet-18 | same cache | same seed | optimizer\nsame schedule | checkpoint selection by id_val loss",
        fill=COLORS["white"],
        edge=COLORS["neutral_edge"],
        title_size=5.25,
        body_size=3.9,
        title_y_pad=0.12,
        body_y_pad=0.35,
    )

    ax.text(2.65, 9.47, "CAMELYON17-WILDS", ha="center", va="center", fontsize=5.7, fontweight="bold", color=COLORS["muted"])
    add_arrow(ax, (5.07, 8.62), (5.15, 8.62))
    add_arrow(ax, (8.22, 8.62), (8.47, 8.62))

    # Stages 4-6: development evidence, freeze, and final one-shot evaluation.
    add_box(
        ax,
        0.25,
        4.10,
        4.10,
        2.55,
        "4  Development evaluation",
        "Full id_val: centers 0, 3, 4\nFull ood_val: center 1\n\nDevelopment comparison only\nCenter 1 is not final held-out evidence\n\nDevelopment evidence selected\nGroupDRO as the predeclared\nprimary candidate",
        fill=COLORS["development_fill"],
        edge=COLORS["development_edge"],
        title_size=6.35,
        body_size=4.75,
        body_y_pad=0.48,
    )

    freeze_body = (
        "Checkpoint paths + SHA256\n"
        "GroupDRO: results/checkpoints/\n"
        "  exp07f_groupdro_resnet18/best.pt\n"
        "  CE0DC65DC6106648F2ABB77C603746A1\n"
        "  DB2B856F9672B01C2E8F540BEEAD8502\n"
        "ERM: results/checkpoints/\n"
        "  exp07f_centerstrat_erm_resnet18/best.pt\n"
        "  8BF990BE517A41AA74D111E0A4F4111A\n"
        "  05A9FE416919261D979F2FD65AE974FF\n"
        "Roles | threshold 0.5 | T=2.974907, 3.496293\n"
        "14 id_val-selected operating thresholds\n"
        "Metrics | high-confidence audit thresholds\n"
        "Output schema | one-run limit"
    )
    add_box(
        ax,
        4.68,
        4.10,
        5.58,
        2.55,
        "5  Frozen before center-2 access",
        freeze_body,
        fill=COLORS["freeze_fill"],
        edge=COLORS["freeze_edge"],
        title_size=6.45,
        body_size=3.55,
        body_y_pad=0.48,
        linewidth=1.8,
        body_family="DejaVu Sans Mono",
    )

    add_box(
        ax,
        10.58,
        4.10,
        4.17,
        2.55,
        "6  One-shot reserved-center\nevaluation",
        "Center 2 | 85,054 patches\nOne dataset instance\nOne dataloader traversal\nBoth models in each batch\nAttempt count = 1\n\nNo calibration refitting\nNo threshold tuning\nNo checkpoint replacement\nNo post-test model selection",
        fill=COLORS["reserved_fill"],
        edge=COLORS["reserved_edge"],
        title_size=5.9,
        body_size=3.95,
        body_y_pad=0.67,
        dashed=True,
        linewidth=1.55,
    )

    add_arrow(ax, (11.55, 7.32), (2.32, 6.69), connectionstyle="angle3,angleA=-90,angleB=180")
    add_arrow(ax, (4.37, 5.38), (4.65, 5.38))
    add_arrow(ax, (10.28, 5.38), (10.55, 5.38))

    # Stage 7: four prespecified analysis categories.
    add_box(
        ax,
        0.25,
        0.76,
        14.50,
        2.62,
        "7  Final analyses",
        fill=COLORS["analysis_fill"],
        edge=COLORS["analysis_edge"],
        title_size=7.2,
        body_y_pad=0.42,
    )
    output_boxes = [
        (
            0.48,
            "1 | Discrimination\nand classification",
            "AUROC | AUPRC | accuracy\nsensitivity | specificity\nconfusion counts",
        ),
        (
            4.03,
            "2 | Calibration",
            "ECE | Brier score | NLL\nraw vs frozen-temperature\ncalibrated",
        ),
        (
            7.58,
            "3 | Operating-point\ntransport",
            "Seven thresholds per model\nnominal vs achieved\nsensitivity / specificity",
        ),
        (
            11.13,
            "4 | High-confidence\nfalse negatives",
            "Confidence 0.90 | 0.95 | 0.99\nraw vs calibrated\nunchanged total false negatives",
        ),
    ]
    for x, title, body in output_boxes:
        add_box(
            ax,
            x,
            1.05,
            3.17,
            1.72,
            title,
            body,
            fill=COLORS["white"],
            edge=COLORS["analysis_edge"],
            title_size=5.0,
            body_size=4.65,
            body_y_pad=0.68 if "\n" in title else 0.46,
        )
    add_arrow(ax, (12.67, 4.07), (12.67, 3.41))

    ax.text(
        7.5,
        0.37,
        "No center-2 refitting, retuning, model switching, or repeat evaluation",
        ha="center",
        va="center",
        fontsize=7.15,
        fontweight="bold",
        color=COLORS["reserved_edge"],
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    svg = OUTPUT_DIR / f"{STEM}.svg"
    pdf = OUTPUT_DIR / f"{STEM}.pdf"
    png = OUTPUT_DIR / f"{STEM}.png"
    common_metadata = {
        "Title": FIGURE_TITLE,
        "Description": "Deterministic schematic of the locked development and reserved-center evaluation protocol.",
        "Creator": "Matplotlib",
    }
    fig.savefig(svg, format="svg", bbox_inches="tight", pad_inches=0.04, metadata={**common_metadata, "Date": None})
    fig.savefig(
        pdf,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.04,
        metadata={
            "Title": FIGURE_TITLE,
            "Subject": common_metadata["Description"],
            "Creator": "Matplotlib",
            "CreationDate": None,
            "ModDate": None,
            "Author": None,
        },
    )
    fig.savefig(
        png,
        format="png",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.04,
        facecolor=COLORS["white"],
        metadata={"Title": FIGURE_TITLE, "Description": common_metadata["Description"], "Author": ""},
    )
    plt.close(fig)
    return {
        "svg": svg.relative_to(ROOT).as_posix(),
        "pdf": pdf.relative_to(ROOT).as_posix(),
        "png": png.relative_to(ROOT).as_posix(),
        "render_dpi": 600,
        "dataset_loaded": False,
        "hf_split_accessed": False,
        "histopathology_image_read": False,
        "checkpoint_accessed": False,
        "model_inference": False,
        "generative_image_model_used": False,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(render(), indent=2))
