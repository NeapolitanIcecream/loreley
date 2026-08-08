"""Render Loreley repository-search graphics as SVG and PNG files."""

from __future__ import annotations

import html
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "docs" / "marketing" / "assets"

W = 1600
H = 900

NAVY = "#0F172A"
SLATE = "#475569"
MUTED = "#64748B"
LINE = "#CBD5E1"
PALE = "#F8FAFC"
WHITE = "#FFFFFF"
CYAN = "#00A9D6"
CYAN_PALE = "#E6F9FD"
VIOLET = "#6563FF"
VIOLET_PALE = "#EFEEFF"
GREEN = "#15803D"
GREEN_PALE = "#EAF8EF"
AMBER = "#B45309"
AMBER_PALE = "#FFF7E6"
RED = "#BE123C"


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def rect(
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    fill: str = WHITE,
    stroke: str = "none",
    radius: float = 24,
    shadow: bool = False,
    stroke_width: float = 2,
) -> str:
    filter_attr = ' filter="url(#shadow)"' if shadow else ""
    return (
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" '
        f'rx="{radius}" fill="{fill}" stroke="{stroke}" '
        f'stroke-width="{stroke_width}"{filter_attr}/>'
    )


def text(
    x: float,
    y: float,
    lines: str | list[str] | tuple[str, ...],
    *,
    size: int = 28,
    fill: str = NAVY,
    weight: int = 400,
    anchor: str = "start",
    line_height: float = 1.25,
    family: str = "Arial, Helvetica, sans-serif",
) -> str:
    if isinstance(lines, str):
        lines = [lines]
    rendered_lines = []
    for index, line in enumerate(lines):
        line_y = y + index * size * line_height
        rendered_lines.append(
            f'<text x="{x}" y="{line_y}" fill="{fill}" font-family="{family}" '
            f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}">'
            f"{esc(line)}</text>"
        )
    return "".join(rendered_lines)


def line(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    stroke: str = LINE,
    width: float = 4,
    dashed: bool = False,
) -> str:
    dash = ' stroke-dasharray="12 10"' if dashed else ""
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
        f'stroke="{stroke}" stroke-width="{width}" stroke-linecap="round"{dash}/>'
    )


def arrow(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    stroke: str = VIOLET,
    width: float = 5,
    dashed: bool = False,
) -> str:
    dash = ' stroke-dasharray="12 10"' if dashed else ""
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
        f'stroke="{stroke}" stroke-width="{width}" stroke-linecap="round" '
        f'marker-end="url(#arrow)"{dash}/>'
    )


def circle(
    cx: float,
    cy: float,
    radius: float,
    *,
    fill: str = WHITE,
    stroke: str = VIOLET,
    stroke_width: float = 4,
) -> str:
    return (
        f'<circle cx="{cx}" cy="{cy}" r="{radius}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{stroke_width}"/>'
    )


def pill(
    x: float,
    y: float,
    label: str,
    *,
    fill: str,
    text_fill: str,
    width: float,
) -> str:
    return "".join(
        [
            rect(x, y, width, 42, fill=fill, radius=21),
            text(x + width / 2, y + 29, label, size=18, fill=text_fill, weight=700, anchor="middle"),
        ]
    )


def svg_document(body: list[str], *, title: str, height: int = H) -> str:
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{height}" viewBox="0 0 {W} {height}">',
            f"<title>{esc(title)}</title>",
            "<defs>",
            '<filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">',
            '<feDropShadow dx="0" dy="8" stdDeviation="12" flood-color="#0F172A" flood-opacity="0.10"/>',
            "</filter>",
            '<marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">',
            '<path d="M0,0 L0,6 L9,3 z" fill="#6563FF"/>',
            "</marker>",
            "</defs>",
            f'<rect width="{W}" height="{height}" fill="{PALE}"/>',
            *body,
            "</svg>",
        ]
    )


def header(title_value: str, subtitle: str) -> list[str]:
    return [
        text(80, 82, title_value, size=48, weight=700),
        text(80, 126, subtitle, size=24, fill=SLATE),
        rect(80, 150, 1440, 4, fill=VIOLET, radius=2),
    ]


def search_loop() -> str:
    body = header(
        "Search the sparse set of useful repository states",
        "Coding agents concentrate proposals; external evaluators decide what survives.",
    )

    cards = [
        (80, "1", "Repository state", ["Complete Git commit", "and recorded ancestry"], CYAN_PALE, CYAN),
        (455, "2", "Coding agents", ["Propose coherent,", "repo-level changes"], VIOLET_PALE, VIOLET),
        (830, "3", "External evaluator", ["Build, verify, score", "and define identity"], GREEN_PALE, GREEN),
        (1205, "4", "QD archive", ["Retain strong, diverse", "states for reuse"], AMBER_PALE, AMBER),
    ]
    for x, number, title_value, detail, fill, accent in cards:
        body.extend(
            [
                rect(x, 245, 315, 250, fill=WHITE, shadow=True),
                circle(x + 52, 298, 25, fill=fill, stroke=accent, stroke_width=3),
                text(x + 52, 307, number, size=23, fill=accent, weight=700, anchor="middle"),
                text(x + 28, 372, title_value, size=28, weight=700),
                text(x + 28, 418, detail, size=22, fill=SLATE, line_height=1.35),
            ]
        )
    for x1, x2 in [(395, 440), (770, 815), (1145, 1190)]:
        body.append(arrow(x1, 370, x2, 370))

    body.extend(
        [
            arrow(1355, 515, 1355, 585, stroke=VIOLET),
            line(1355, 585, 250, 585, stroke=VIOLET, width=5),
            arrow(250, 585, 250, 515, stroke=VIOLET),
            text(805, 621, "Successful states become parents or inspirations", size=21, fill=VIOLET, weight=700, anchor="middle"),
            rect(80, 684, 690, 140, fill=CYAN_PALE, radius=22),
            text(112, 730, "The agent changes the proposal distribution", size=25, fill=NAVY, weight=700),
            text(112, 770, ["It uses repository semantics and prior feedback instead of", "sampling arbitrary syntax."], size=20, fill=SLATE, line_height=1.35),
            rect(830, 684, 690, 140, fill=GREEN_PALE, radius=22),
            text(862, 730, "The evaluator controls archive admission", size=25, fill=NAVY, weight=700),
            text(862, 770, ["Only candidates that build, remain correct, and improve the", "frozen objective can enter the archive."], size=20, fill=SLATE, line_height=1.35),
        ]
    )
    return svg_document(body, title="Loreley repository search loop")


def case_card(
    *,
    x: int,
    name: str,
    value: str,
    metric: str,
    status: str,
    status_fill: str,
    status_text: str,
    facts: list[str],
    scope_note: list[str],
) -> list[str]:
    result = [
        rect(x, 205, 450, 600, fill=WHITE, shadow=True),
        rect(x, 205, 450, 10, fill=VIOLET, radius=5),
        text(x + 30, 270, name, size=30, weight=700),
        text(x + 30, 348, value, size=58, fill=VIOLET, weight=700),
        text(x + 32, 388, metric, size=20, fill=SLATE),
        pill(x + 30, 420, status, fill=status_fill, text_fill=status_text, width=300),
        text(x + 30, 505, "Run summary", size=21, fill=MUTED, weight=700),
    ]
    for index, fact in enumerate(facts):
        result.extend(
            [
                circle(x + 43, 548 + index * 45, 7, fill=CYAN, stroke=CYAN, stroke_width=1),
                text(x + 64, 556 + index * 45, fact, size=20, fill=NAVY),
            ]
        )
    result.extend(
        [
            line(x + 30, 688, x + 420, 688, stroke=LINE, width=2),
            text(x + 30, 728, "Scope note", size=21, fill=MUTED, weight=700),
            text(x + 30, 762, scope_note, size=18, fill=SLATE, line_height=1.3),
        ]
    )
    return result


def three_case_evidence() -> str:
    body = header(
        "Results from three repository searches",
        "Selection protocol and workload scope are shown with each result.",
    )
    body.extend(
        case_card(
            x=80,
            name="markdown-it-py",
            value="+6.75%",
            metric="throughput · separate 28-doc corpus",
            status="Prospective result",
            status_fill=GREEN_PALE,
            status_text=GREEN,
            facts=["64 jobs · generation 4", "28 / 28 documents improved", "Winner frozen before validation"],
            scope_note=["One repository and host;", "human-written seeds."],
        )
    )
    body.extend(
        case_card(
            x=575,
            name="python-pathspec",
            value="+25.14%",
            metric="throughput · five reference workloads",
            status="Post-hoc selection",
            status_fill=AMBER_PALE,
            status_text=AMBER,
            facts=["64 jobs · generation 4", "5 / 5 workloads improved", "Archive revisited a retained branch"],
            scope_note=["Candidate selected after the", "allocation gate was revealed."],
        )
    )
    body.extend(
        case_card(
            x=1070,
            name="Zstandard V19",
            value="+1.019%",
            metric="compression · sealed holdout",
            status="Preregistered result",
            status_fill=VIOLET_PALE,
            status_text=VIOLET,
            facts=["220 jobs · 167 binaries", "95% CI +0.962% to +1.076%", "Top-10 follow-up: +0.891%"],
            scope_note=["Registered winner is a manual seed;", "follow-up used a different corpus."],
        )
    )
    body.append(text(800, 850, "348 terminal jobs · 310 successful · 38 failed · fixed-repository results", size=21, fill=SLATE, weight=700, anchor="middle"))
    return svg_document(body, title="Loreley three-case evidence summary")


def node(
    body: list[str],
    *,
    x: float,
    y: float,
    generation: str,
    title_value: str,
    metric: str,
    accent: str,
) -> None:
    body.extend(
        [
            circle(x, y, 33, fill=WHITE, stroke=accent, stroke_width=5),
            text(x, y + 8, generation, size=20, fill=accent, weight=700, anchor="middle"),
            text(x, y + 70, title_value, size=18, fill=NAVY, weight=700, anchor="middle"),
            text(x, y + 98, metric, size=17, fill=MUTED, anchor="middle"),
        ]
    )


def lineages() -> str:
    body = header(
        "Useful lineages can accumulate compatible changes",
        "Recorded ancestry shows how compatible changes accumulated across generations.",
    )

    body.extend(
        [
            rect(80, 205, 1440, 270, fill=WHITE, shadow=True),
            pill(112, 235, "markdown-it-py", fill=CYAN_PALE, text_fill=CYAN, width=220),
            text(360, 264, "Independent validation: +6.75%", size=22, fill=GREEN, weight=700),
            line(200, 335, 1400, 335, stroke=LINE, width=5),
        ]
    )
    markdown_nodes = [
        (210, "G1", "Inline HTML", "job 7"),
        (600, "G2", "Renderer + attrs", "job 12"),
        (990, "G3", "Escape + dispatch", "job 14"),
        (1380, "G4", "Normalization", "job 26 · winner"),
    ]
    for x, gen, title_value, metric in markdown_nodes:
        node(body, x=x, y=335, generation=gen, title_value=title_value, metric=metric, accent=CYAN)
    for x1, x2 in [(248, 562), (638, 952), (1028, 1342)]:
        body.append(arrow(x1, 335, x2, 335, stroke=CYAN, width=4))

    body.extend(
        [
            rect(80, 515, 1440, 300, fill=WHITE, shadow=True),
            pill(112, 545, "python-pathspec", fill=VIOLET_PALE, text_fill=VIOLET, width=220),
            text(360, 574, "Reference workloads: +25.14% · post-hoc selection", size=22, fill=AMBER, weight=700),
            line(175, 665, 1410, 665, stroke=LINE, width=5),
        ]
    )
    pathspec_nodes = [
        (180, "G0", "C-level iterators", "0.9978×"),
        (455, "G1", "Bind hot calls", "1.0721×"),
        (730, "G2", "Remove groupdict", "1.0866×"),
        (1005, "G3", "Direct regex search", "1.1921×"),
        (1400, "G4", "Flatten dispatch", "1.2536× training"),
    ]
    for x, gen, title_value, metric in pathspec_nodes:
        node(body, x=x, y=665, generation=gen, title_value=title_value, metric=metric, accent=VIOLET)
    for x1, x2 in [(218, 417), (493, 692), (768, 967)]:
        body.append(arrow(x1, 665, x2, 665, stroke=VIOLET, width=4))
    body.extend(
        [
            arrow(1043, 665, 1362, 665, stroke=VIOLET, width=4, dashed=True),
            pill(1080, 610, "20 other jobs explored elsewhere", fill=AMBER_PALE, text_fill=AMBER, width=270),
            text(800, 864, "Parent ancestry and inspiration edges are recorded separately.", size=20, fill=SLATE, anchor="middle"),
        ]
    )
    return svg_document(body, title="Loreley case-study lineages")


def interval_plot(
    body: list[str],
    *,
    y: float,
    label: str,
    estimate: float,
    low: float,
    high: float,
    color: str,
    note: str,
) -> None:
    chart_x = 720
    chart_w = 760
    max_value = 1.5

    def scale(value: float) -> float:
        return chart_x + chart_w * value / max_value

    body.extend(
        [
            text(120, y - 10, label, size=24, fill=NAVY, weight=700),
            text(120, y + 25, note, size=18, fill=MUTED),
            line(chart_x, y, chart_x + chart_w, y, stroke=LINE, width=4),
            line(scale(low), y, scale(high), y, stroke=color, width=10),
            line(scale(low), y - 13, scale(low), y + 13, stroke=color, width=4),
            line(scale(high), y - 13, scale(high), y + 13, stroke=color, width=4),
            circle(scale(estimate), y, 12, fill=WHITE, stroke=color, stroke_width=6),
            text(scale(estimate), y - 27, f"{estimate:.3f}%", size=18, fill=color, weight=700, anchor="middle"),
        ]
    )


def zstd_identity_results() -> str:
    body = header(
        "Zstandard V19: source identity, binary identity, measured effect",
        "A Git commit records ancestry; the evaluator decides whether a performance state is new.",
    )
    identity_y = 220
    identity_cards = [
        (80, 300, "220", "terminal jobs", VIOLET_PALE, VIOLET),
        (450, 300, "211", "successful source states", CYAN_PALE, CYAN),
        (820, 300, "167", "distinct release binaries", GREEN_PALE, GREEN),
        (1190, 330, "19", "cached repeat binaries", AMBER_PALE, AMBER),
    ]
    for x, width, value, label, fill, accent in identity_cards:
        body.extend(
            [
                rect(x, identity_y, width, 150, fill=fill, radius=24),
                text(x + 28, identity_y + 64, value, size=48, fill=accent, weight=700),
                text(x + 28, identity_y + 108, label, size=20, fill=NAVY, weight=700),
            ]
        )
    for x1, x2 in [(395, 435), (765, 805), (1135, 1175)]:
        body.append(arrow(x1, identity_y + 75, x2, identity_y + 75))
    body.extend(
        [
            text(980, 407, "44 successful jobs repeated an existing binary", size=18, fill=MUTED, anchor="middle"),
            text(980, 435, "25 measured before cache · 19 reused after cache", size=18, fill=MUTED, anchor="middle"),
            rect(80, 480, 1440, 330, fill=WHITE, shadow=True),
            text(120, 535, "Compression throughput gain with 95% confidence intervals", size=28, weight=700),
        ]
    )

    chart_x = 720
    chart_w = 760
    for value in (0.0, 0.5, 1.0, 1.5):
        x = chart_x + chart_w * value / 1.5
        body.extend(
            [
                line(x, 570, x, 745, stroke="#E2E8F0", width=2),
                text(x, 775, f"{value:.1f}%", size=17, fill=MUTED, anchor="middle"),
            ]
        )
    interval_plot(
        body,
        y=615,
        label="Registered winner",
        estimate=1.019,
        low=0.962,
        high=1.076,
        color=VIOLET,
        note="manual seed · sealed holdout",
    )
    interval_plot(
        body,
        y=705,
        label="Top-10 follow-up",
        estimate=0.891,
        low=0.522,
        high=1.261,
        color=CYAN,
        note="generation 4 · new disjoint corpus",
    )
    body.extend(
        [
            pill(80, 835, "Different fresh corpora · not directly comparable", fill=AMBER_PALE, text_fill=AMBER, width=500),
            text(1520, 864, "Registered winner: manual seed · +1.019%", size=20, fill=SLATE, weight=700, anchor="end"),
        ]
    )
    return svg_document(body, title="Loreley Zstandard identity and result summary")


ASSETS = {
    "loreley-search-loop": search_loop,
    "loreley-three-case-evidence": three_case_evidence,
    "loreley-case-lineages": lineages,
    "loreley-zstd-identity-results": zstd_identity_results,
}


def render() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for stem, builder in ASSETS.items():
        svg_path = OUTPUT / f"{stem}.svg"
        png_path = OUTPUT / f"{stem}.png"
        svg_path.write_text(builder(), encoding="utf-8")
        subprocess.run(
            ["sips", "-s", "format", "png", str(svg_path), "--out", str(png_path)],
            check=True,
            stdout=subprocess.DEVNULL,
        )


if __name__ == "__main__":
    render()
