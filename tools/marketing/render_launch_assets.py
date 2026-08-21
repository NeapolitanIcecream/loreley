"""Render Loreley repository-search graphics as SVG and PNG files."""

from __future__ import annotations

import html
import subprocess
from pathlib import Path
from typing import TypedDict, Unpack

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


class RectOptions(TypedDict, total=False):
    fill: str
    stroke: str
    radius: float
    shadow: bool
    stroke_width: float


class TextOptions(TypedDict, total=False):
    size: int
    fill: str
    weight: int
    anchor: str
    line_height: float
    family: str


class LineOptions(TypedDict, total=False):
    stroke: str
    width: float
    dashed: bool


class CaseCardOptions(TypedDict):
    x: int
    name: str
    value: str
    metric: str
    status: str
    status_fill: str
    status_text: str
    facts: list[str]
    scope_note: list[str]


class NodeOptions(TypedDict):
    x: float
    y: float
    generation: str
    title_value: str
    metric: str
    accent: str


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def rect(
    x: float,
    y: float,
    width: float,
    height: float,
    **options: Unpack[RectOptions],
) -> str:
    fill = options.get("fill", WHITE)
    stroke = options.get("stroke", "none")
    radius = options.get("radius", 24)
    shadow = options.get("shadow", False)
    stroke_width = options.get("stroke_width", 2)
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
    **options: Unpack[TextOptions],
) -> str:
    size = options.get("size", 28)
    fill = options.get("fill", NAVY)
    weight = options.get("weight", 400)
    anchor = options.get("anchor", "start")
    line_height = options.get("line_height", 1.25)
    family = options.get("family", "Arial, Helvetica, sans-serif")
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
    **options: Unpack[LineOptions],
) -> str:
    stroke = options.get("stroke", LINE)
    width = options.get("width", 4)
    dashed = options.get("dashed", False)
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
    **options: Unpack[LineOptions],
) -> str:
    stroke = options.get("stroke", VIOLET)
    width = options.get("width", 5)
    dashed = options.get("dashed", False)
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
            text(
                x + width / 2,
                y + 29,
                label,
                size=18,
                fill=text_fill,
                weight=700,
                anchor="middle",
            ),
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
        (
            80,
            "1",
            "Repository state",
            ["Complete Git commit", "and recorded ancestry"],
            CYAN_PALE,
            CYAN,
        ),
        (
            455,
            "2",
            "Coding agents",
            ["Propose coherent,", "repo-level changes"],
            VIOLET_PALE,
            VIOLET,
        ),
        (
            830,
            "3",
            "External evaluator",
            ["Build, verify, score", "and define identity"],
            GREEN_PALE,
            GREEN,
        ),
        (
            1205,
            "4",
            "QD archive",
            ["Retain strong, diverse", "states for reuse"],
            AMBER_PALE,
            AMBER,
        ),
    ]
    for x, number, title_value, detail, fill, accent in cards:
        body.extend(
            [
                rect(x, 245, 315, 250, fill=WHITE, shadow=True),
                circle(x + 52, 298, 25, fill=fill, stroke=accent, stroke_width=3),
                text(
                    x + 52,
                    307,
                    number,
                    size=23,
                    fill=accent,
                    weight=700,
                    anchor="middle",
                ),
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
            text(
                805,
                621,
                "Successful states become parents or inspirations",
                size=21,
                fill=VIOLET,
                weight=700,
                anchor="middle",
            ),
            rect(80, 684, 690, 140, fill=CYAN_PALE, radius=22),
            text(
                112,
                730,
                "The agent changes the proposal distribution",
                size=25,
                fill=NAVY,
                weight=700,
            ),
            text(
                112,
                770,
                [
                    "It uses repository semantics and prior feedback instead of",
                    "sampling arbitrary syntax.",
                ],
                size=20,
                fill=SLATE,
                line_height=1.35,
            ),
            rect(830, 684, 690, 140, fill=GREEN_PALE, radius=22),
            text(
                862,
                730,
                "The evaluator controls archive admission",
                size=25,
                fill=NAVY,
                weight=700,
            ),
            text(
                862,
                770,
                [
                    "Only candidates that build, remain correct, and improve the",
                    "frozen objective can enter the archive.",
                ],
                size=20,
                fill=SLATE,
                line_height=1.35,
            ),
        ]
    )
    return svg_document(body, title="Loreley repository search loop")


def case_card(**options: Unpack[CaseCardOptions]) -> list[str]:
    x = options["x"]
    name = options["name"]
    value = options["value"]
    metric = options["metric"]
    status = options["status"]
    status_fill = options["status_fill"]
    status_text = options["status_text"]
    facts = options["facts"]
    scope_note = options["scope_note"]
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
                circle(
                    x + 43, 548 + index * 45, 7, fill=CYAN, stroke=CYAN, stroke_width=1
                ),
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
            facts=[
                "64 jobs · generation 4",
                "28 / 28 documents improved",
                "Winner frozen before validation",
            ],
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
            facts=[
                "64 jobs · generation 4",
                "5 / 5 workloads improved",
                "Archive revisited a retained branch",
            ],
            scope_note=[
                "Candidate selected after the",
                "allocation gate was revealed.",
            ],
        )
    )
    body.extend(
        case_card(
            x=1070,
            name="Zstandard",
            value="+1.173%",
            metric="compression · original holdout",
            status="Validation-selected",
            status_fill=AMBER_PALE,
            status_text=AMBER,
            facts=[
                "fe39bee8 · generation 4",
                "Fresh sealed corpus: +0.891%",
                "10 / 10 fixed finalists positive",
            ],
            scope_note=[
                "Holdout opened at study level;",
                "fresh recipe chosen post-fix.",
            ],
        )
    )
    body.append(
        text(
            800,
            850,
            "348 terminal jobs · 310 successful · 38 failed · fixed-repository results",
            size=21,
            fill=SLATE,
            weight=700,
            anchor="middle",
        )
    )
    return svg_document(body, title="Loreley three-case evidence summary")


def node(
    body: list[str],
    **options: Unpack[NodeOptions],
) -> None:
    x = options["x"]
    y = options["y"]
    generation = options["generation"]
    title_value = options["title_value"]
    metric = options["metric"]
    accent = options["accent"]
    body.extend(
        [
            circle(x, y, 33, fill=WHITE, stroke=accent, stroke_width=5),
            text(
                x, y + 8, generation, size=20, fill=accent, weight=700, anchor="middle"
            ),
            text(
                x, y + 70, title_value, size=18, fill=NAVY, weight=700, anchor="middle"
            ),
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
            text(
                360,
                264,
                "Independent validation: +6.75%",
                size=22,
                fill=GREEN,
                weight=700,
            ),
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
        node(
            body,
            x=x,
            y=335,
            generation=gen,
            title_value=title_value,
            metric=metric,
            accent=CYAN,
        )
    for x1, x2 in [(248, 562), (638, 952), (1028, 1342)]:
        body.append(arrow(x1, 335, x2, 335, stroke=CYAN, width=4))

    body.extend(
        [
            rect(80, 515, 1440, 300, fill=WHITE, shadow=True),
            pill(
                112,
                545,
                "python-pathspec",
                fill=VIOLET_PALE,
                text_fill=VIOLET,
                width=220,
            ),
            text(
                360,
                574,
                "Reference workloads: +25.14% · post-hoc selection",
                size=22,
                fill=AMBER,
                weight=700,
            ),
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
        node(
            body,
            x=x,
            y=665,
            generation=gen,
            title_value=title_value,
            metric=metric,
            accent=VIOLET,
        )
    for x1, x2 in [(218, 417), (493, 692), (768, 967)]:
        body.append(arrow(x1, 665, x2, 665, stroke=VIOLET, width=4))
    body.extend(
        [
            arrow(1043, 665, 1362, 665, stroke=VIOLET, width=4, dashed=True),
            pill(
                1080,
                610,
                "20 other jobs explored elsewhere",
                fill=AMBER_PALE,
                text_fill=AMBER,
                width=270,
            ),
            text(
                800,
                864,
                "Parent ancestry and inspiration edges are recorded separately.",
                size=20,
                fill=SLATE,
                anchor="middle",
            ),
        ]
    )
    return svg_document(body, title="Loreley case-study lineages")


def zstd_identity_flow(identity_y: int) -> list[str]:
    body: list[str] = []
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
            text(
                980,
                407,
                "44 successful jobs repeated an existing binary",
                size=18,
                fill=MUTED,
                anchor="middle",
            ),
            text(
                980,
                435,
                "25 measured before cache · 19 reused after cache",
                size=18,
                fill=MUTED,
                anchor="middle",
            ),
        ]
    )
    return body


def zstd_holdout_leader() -> list[str]:
    return [
        rect(80, 480, 1440, 330, fill=WHITE, shadow=True),
        text(
            120,
            535,
            "Descriptive holdout leader",
            size=28,
            weight=700,
        ),
        line(820, 525, 820, 780, stroke=LINE, width=2),
        text(
            120,
            585,
            "5ee53426 · generation 3",
            size=22,
            fill=SLATE,
            weight=700,
        ),
        text(
            120,
            675,
            "+1.228%",
            size=68,
            fill=VIOLET,
            weight=700,
        ),
        text(
            120,
            720,
            "compression throughput · original holdout",
            size=21,
            fill=SLATE,
        ),
        text(
            120,
            758,
            "95% CI: +1.125% to +1.330%",
            size=20,
            fill=MUTED,
        ),
        text(
            120,
            788,
            "Ranked by the compression lower 95% bound",
            size=18,
            fill=MUTED,
        ),
    ]


def zstd_top10_comparison() -> list[str]:
    return [
        text(
            870,
            535,
            "Fixed Top-10 comparison",
            size=26,
            fill=NAVY,
            weight=700,
        ),
        text(
            870,
            625,
            "10 / 10",
            size=50,
            fill=CYAN,
            weight=700,
        ),
        text(
            870,
            665,
            "positive compression results",
            size=20,
            fill=SLATE,
        ),
        text(
            870,
            715,
            "Median gain: +1.116%",
            size=22,
            fill=NAVY,
            weight=700,
        ),
        text(
            870,
            752,
            "Point range: +0.856% to +1.239%",
            size=19,
            fill=SLATE,
        ),
        text(
            870,
            785,
            "Runner-up fe39bee8: +1.173%",
            size=18,
            fill=MUTED,
        ),
    ]


def zstd_result_footer() -> list[str]:
    return [
        pill(
            80,
            835,
            "Post-selection sensitivity · not a new blinded winner",
            fill=AMBER_PALE,
            text_fill=AMBER,
            width=600,
        ),
        text(
            1520,
            864,
            "Fresh-corpus result for fe39bee8: +0.891% (95% CI +0.522% to +1.261%)",
            size=18,
            fill=SLATE,
            weight=700,
            anchor="end",
        ),
    ]


def zstd_identity_results() -> str:
    body = header(
        "Zstandard: Top-10 holdout and binary identity",
        "5ee53426 led descriptively; all ten fixed candidates remained positive.",
    )
    body.extend(zstd_identity_flow(identity_y=220))
    body.extend(zstd_holdout_leader())
    body.extend(zstd_top10_comparison())
    body.extend(zstd_result_footer())
    return svg_document(
        body, title="Loreley Zstandard Top-10 holdout and identity summary"
    )


def paper_overview_header() -> list[str]:
    return [
        pill(
            80,
            42,
            "LORELEY",
            fill=VIOLET_PALE,
            text_fill=VIOLET,
            width=150,
        ),
        text(
            250,
            71,
            "Repository-scale program evolution with Quality-Diversity search",
            size=34,
            weight=700,
        ),
        text(
            80,
            122,
            "Coding agents edit complete Git states; external evaluators build, verify, and score every candidate.",
            size=22,
            fill=SLATE,
        ),
        pill(
            1270,
            94,
            "arXiv:2608.19703",
            fill=CYAN_PALE,
            text_fill=CYAN,
            width=250,
        ),
        rect(80, 150, 1440, 4, fill=VIOLET, radius=2),
        text(80, 190, "WHAT LORELEY SEARCHES", size=17, fill=MUTED, weight=700),
    ]


def paper_overview_method_step(
    x: int,
    number: str,
    title_value: str,
    detail: str,
    fill: str,
    accent: str,
) -> list[str]:
    return [
        rect(x, 210, 315, 84, fill=WHITE, shadow=True, radius=18),
        circle(x + 36, 252, 20, fill=fill, stroke=accent, stroke_width=3),
        text(
            x + 36,
            259,
            number,
            size=18,
            fill=accent,
            weight=700,
            anchor="middle",
        ),
        text(x + 68, 246, title_value, size=21, weight=700),
        text(x + 68, 275, detail, size=16, fill=MUTED),
    ]


def paper_overview_method() -> list[str]:
    body: list[str] = []
    method_steps = [
        (80, "1", "Complete Git state", "source + ancestry", CYAN_PALE, CYAN),
        (455, "2", "Coding agent", "repository-level edit", VIOLET_PALE, VIOLET),
        (830, "3", "External evaluator", "build + verify + score", GREEN_PALE, GREEN),
        (1205, "4", "QD archive", "retain + resample", AMBER_PALE, AMBER),
    ]
    for step in method_steps:
        body.extend(paper_overview_method_step(*step))
    for x1, x2 in [(400, 440), (775, 815), (1150, 1190)]:
        body.append(arrow(x1, 252, x2, 252, stroke=VIOLET, width=4))
    return body


def paper_overview_matched_header() -> list[str]:
    return [
        rect(80, 326, 1440, 340, fill=WHITE, shadow=True, radius=22),
        text(
            112,
            365,
            "MATCHED ZSTANDARD COMPARISON",
            size=17,
            fill=MUTED,
            weight=700,
        ),
        text(
            112,
            407,
            "1,008 candidate jobs",
            size=32,
            fill=NAVY,
            weight=700,
        ),
        text(
            455,
            405,
            "7 paired blocks · 48 jobs per policy and block · held-out compression throughput",
            size=19,
            fill=SLATE,
        ),
    ]


def paper_overview_policy_card(
    x: int,
    policy: str,
    mean_value: str,
    detail: str,
    fill: str,
    accent: str,
) -> list[str]:
    return [
        rect(x, 433, 420, 128, fill=fill, radius=18),
        text(x + 24, 465, policy, size=20, fill=NAVY, weight=700),
        text(x + 24, 514, mean_value, size=38, fill=accent, weight=700),
        text(x + 198, 509, "mean", size=16, fill=MUTED, weight=700),
        text(x + 24, 544, detail, size=16, fill=SLATE),
    ]


def paper_overview_policy_cards() -> list[str]:
    body: list[str] = []
    policy_cards = [
        (
            112,
            "Independent Root",
            "+0.502%",
            "+0.412% median · 2/7 ≥ +0.5%",
            CYAN_PALE,
            CYAN,
        ),
        (
            580,
            "Loreley QD",
            "+0.824%",
            "+0.739% median · 6/7 ≥ +0.5%",
            VIOLET_PALE,
            VIOLET,
        ),
        (
            1048,
            "Sequential Champion",
            "+0.960%",
            "+0.819% median · 7/7 ≥ +0.5%",
            GREEN_PALE,
            GREEN,
        ),
    ]
    for card in policy_cards:
        body.extend(paper_overview_policy_card(*card))
    body.append(
        text(
            1444,
            463,
            "HIGHEST OBSERVED",
            size=13,
            fill=GREEN,
            weight=700,
            anchor="end",
        )
    )
    return body


def paper_overview_outcome_card(
    x: int,
    width: int,
    title_value: str,
    detail: str,
    fill: str,
    accent: str,
) -> list[str]:
    return [
        rect(x, 583, width, 60, fill=fill, radius=16),
        text(x + 24, 608, title_value, size=15, fill=accent, weight=700),
        text(x + 24, 631, detail, size=18, fill=NAVY, weight=700),
    ]


def paper_overview_outcomes() -> list[str]:
    body = paper_overview_outcome_card(
        112,
        660,
        "ARCHIVE ACTIVITY · 7/7 BLOCKS",
        "Retained alternatives were sampled again.",
        VIOLET_PALE,
        VIOLET,
    )
    body.extend(
        paper_overview_outcome_card(
            796,
            692,
            "48-JOB ENDPOINT",
            "QD did not establish superiority over either control.",
            AMBER_PALE,
            AMBER,
        )
    )
    return body


def paper_overview_capability_card(
    x: int, name: str, value: str, detail: str, accent: str
) -> list[str]:
    return [
        rect(x, 724, 450, 112, fill=WHITE, shadow=True, radius=18),
        rect(x, 724, 7, 112, fill=accent, radius=3),
        text(x + 26, 756, name, size=19, fill=NAVY, weight=700),
        text(x + 26, 798, value, size=31, fill=accent, weight=700),
        text(x + 26, 823, detail, size=15, fill=SLATE),
    ]


def paper_overview_capabilities() -> list[str]:
    body = [
        text(
            80,
            704,
            "EARLIER CAPABILITY CAMPAIGNS · FIXED REPOSITORIES, DIFFERENT PROTOCOLS",
            size=17,
            fill=MUTED,
            weight=700,
        )
    ]
    capability_cards = [
        (
            80,
            "markdown-it-py",
            "+6.75%",
            "separate 28-document validation",
            CYAN,
        ),
        (
            575,
            "python-pathspec",
            "+25.14%",
            "post-hoc selection · five workloads",
            VIOLET,
        ),
        (
            1070,
            "Zstandard · evolved candidate",
            "+1.173% H  ·  +0.891% F",
            "validation-selected · protocol limits apply",
            GREEN,
        ),
    ]
    for card in capability_cards:
        body.extend(paper_overview_capability_card(*card))
    return body


def paper_overview_footer() -> list[str]:
    return [
        text(
            80,
            874,
            "Capability results do not estimate average gains on new repositories. H = original holdout; F = fresh sealed corpus.",
            size=16,
            fill=MUTED,
        ),
        text(
            1520,
            874,
            "github.com/NeapolitanIcecream/loreley",
            size=16,
            fill=VIOLET,
            weight=700,
            anchor="end",
        ),
    ]


def paper_overview() -> str:
    """Render the arXiv launch card for the matched-policy paper."""

    body = paper_overview_header()
    body.extend(paper_overview_method())
    body.extend(paper_overview_matched_header())
    body.extend(paper_overview_policy_cards())
    body.extend(paper_overview_outcomes())
    body.extend(paper_overview_capabilities())
    body.extend(paper_overview_footer())
    return svg_document(body, title="Loreley paper overview")


ASSETS = {
    "loreley-search-loop": search_loop,
    "loreley-three-case-evidence": three_case_evidence,
    "loreley-case-lineages": lineages,
    "loreley-zstd-identity-results": zstd_identity_results,
    "loreley-paper-overview": paper_overview,
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
