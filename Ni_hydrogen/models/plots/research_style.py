# Copied from the research-plots skill (assets/research_style.py, plugin version 1.0.0) on 2026-11-24. Do not edit this copy.
"""The research plotting style for matplotlib, and a check for a finished figure.

The values come from the figure the user approved on 2026-11-06 ("Keep this version"; a local,
unpublished copy is approved_example.png beside this file): 180 mm wide, TeX Gyre Heros, 17 pt tick labels, 20 pt axis
titles, 2.2 pt black frame and ticks (inside, four sides), 4 pt data outlines, all text black,
Okabe-Ito colours, white background, no grid, PDF and 600 dpi PNG.

Copy this file into the project's figure folder and import the copy, so that figure scripts
rebuild without the skill installed:

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from research_style import use_research_style, save, check_figure, COLOURS, MM

    use_research_style()
    fig, ax = plt.subplots(figsize=(180 * MM, 128 * MM))
    ...
    for problem in check_figure(fig):
        print(problem)
    save(fig, "figures/fig_name")   # fig_name.pdf and fig_name.png
"""
import colorsys
import glob
import pathlib
import re
import subprocess

import matplotlib as mpl
from matplotlib import font_manager
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.text import Text

MM = 1 / 25.4                 # inches per millimetre; figsize=(180 * MM, 128 * MM)

TICK_PT = 17                  # tick labels
LABEL_PT = 20                 # axis titles
ANNOT_PT = 16                 # direct labels inside the plot, bold
MIN_TEXT_PT = 14              # check_figure flags visible text smaller than this

FRAME_PT = 2.2                # axes frame and major ticks
MINOR_PT = 1.5                # minor ticks
OUTLINE_PT = 4.0              # boxes and outlines, as in the approved figure
CURVE_PT = 3.0                # default curve width: thicker than the frame; use OUTLINE_PT for 3 curves or fewer

# Okabe-Ito, full strength. Purple is darkened for thick lines and labels (approved figure).
OKABE_ITO = {
    "black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
    "yellow": "#F0E442", "blue": "#0072B2", "vermilion": "#D55E00", "purple": "#CC79A7",
}
COLOURS = {
    "black": "#000000",
    "blue": "#0072B2",
    "vermilion": "#D55E00",
    "green": "#009E73",
    "purple": "#B4508C",      # darker than Okabe-Ito #CC79A7 so a thick line and its label read well
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "grey_points": "#5A5A5A", # data marks that step back; never text
}
# Yellow is left out of the cycle: it is too light for a line or text on white. Use it only as a fill
# with a black outline and a hatch pattern.
CYCLE = [COLOURS[k] for k in ("black", "blue", "vermilion", "green", "purple", "orange", "sky")]

FONT_CHOICES = ["TeX Gyre Heros", "Helvetica", "Arial"]


def _tex_gyre_heros_files():
    """TeX Gyre Heros from the TeX installation: kpsewhich first, then the usual TeX Live folders."""
    names = [f"texgyreheros-{s}.otf" for s in ("regular", "bold", "italic", "bolditalic")]
    try:
        found = subprocess.run(["kpsewhich", *names], capture_output=True, text=True, timeout=30).stdout
        paths = [p for p in found.splitlines() if p.strip()]
    except (OSError, subprocess.SubprocessError):
        paths = []
    if not paths:
        tail = "texmf-dist/fonts/opentype/public/tex-gyre/texgyreheros-*.otf"
        for root in ("/usr/local/texlive/*", "/usr/share/texlive", "/opt/texlive/*"):   # TeX Live, Debian
            paths += glob.glob(f"{root}/{tail}")
    return paths


def register_fonts():
    """Add TeX Gyre Heros if TeX is installed. Return the first face of FONT_CHOICES found."""
    for path in _tex_gyre_heros_files():
        font_manager.fontManager.addfont(path)
    known = {f.name for f in font_manager.fontManager.ttflist}
    for name in FONT_CHOICES:
        if name in known:
            return name
    return "DejaVu Sans"


def use_research_style():
    """Set the rcParams. Returns the typeface in use."""
    face = register_fonts()
    mpl.rcParams.update({
        # surface and ink
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "text.color": "black", "axes.labelcolor": "black", "axes.edgecolor": "black",
        "xtick.color": "black", "ytick.color": "black", "xtick.labelcolor": "black", "ytick.labelcolor": "black",
        # type: one face for text and math
        "font.family": "sans-serif", "font.sans-serif": [face, *FONT_CHOICES, "DejaVu Sans"],
        "mathtext.fontset": "custom", "mathtext.rm": f"{face}:regular", "mathtext.it": f"{face}:italic",
        "mathtext.bf": f"{face}:bold",  # math letters stay italic: $n$ = 10,535
        "font.size": TICK_PT, "axes.labelsize": LABEL_PT, "axes.titlesize": LABEL_PT, "axes.titleweight": "bold",
        "xtick.labelsize": TICK_PT, "ytick.labelsize": TICK_PT,
        "legend.fontsize": ANNOT_PT, "legend.frameon": False, "legend.handlelength": 2.4,
        # frame and ticks: black, thick, inside, on all four sides
        "axes.linewidth": FRAME_PT, "axes.grid": False,
        "axes.spines.top": True, "axes.spines.right": True,
        "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
        "xtick.major.width": FRAME_PT, "ytick.major.width": FRAME_PT,
        "xtick.minor.width": MINOR_PT, "ytick.minor.width": MINOR_PT,
        "xtick.major.size": 9, "ytick.major.size": 9, "xtick.minor.size": 5, "ytick.minor.size": 5,
        "xtick.major.pad": 7, "ytick.major.pad": 7,
        "axes.formatter.use_mathtext": True,
        # data
        "axes.prop_cycle": mpl.cycler(color=CYCLE),
        "lines.linewidth": CURVE_PT, "lines.markersize": 8, "lines.markeredgewidth": 1.5,
        "lines.solid_capstyle": "round", "lines.dash_capstyle": "round",
        "patch.linewidth": 1.2, "patch.edgecolor": "black", "hatch.linewidth": 1.2, "hatch.color": "black",
        "errorbar.capsize": 4, "scatter.edgecolors": "none",
        # layout keeps the figure at the size set in figsize, so the text prints at its stated size
        "figure.constrained_layout.use": True,
        # output: vector PDF with the font embedded and text kept as text, 600 dpi raster
        "figure.dpi": 150, "savefig.dpi": 600, "savefig.bbox": None,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    })
    return face


def save(fig, stem, formats=("pdf", "png")):
    """Write stem.pdf and stem.png at 600 dpi on white. Returns the paths."""
    stem = pathlib.Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in formats:
        path = stem.with_suffix(f".{ext}")
        fig.savefig(path, dpi=600, facecolor="white")
        paths.append(path)
    return paths


def _is_grey(colour):
    r, g, b = to_rgb(colour)
    _, lightness, saturation = colorsys.rgb_to_hls(r, g, b)
    return saturation < 0.12 and lightness > 0.15


def _not_white(colour):
    return to_rgb(colour) != (1.0, 1.0, 1.0)


def _undrawn_tick_labels(fig):
    """Tick labels of ticks outside the view. matplotlib keeps them but does not draw them."""
    hidden = set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            lo, hi = sorted(axis.get_view_interval())
            tol = 1e-9 * max(abs(lo), abs(hi), 1.0)
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                loc = tick.get_loc()
                if loc is None or not lo - tol <= loc <= hi + tol:
                    hidden.update({id(tick.label1), id(tick.label2)})
    return hidden


def check_figure(fig, min_text_pt=MIN_TEXT_PT):
    """List the places where a figure breaks the specification. An empty list means none found.

    It checks what a script can measure: text size, grey text, typeface, text that runs off the
    figure, background, frame, data line widths, grid lines and legends. It cannot judge clutter,
    overlaps or colour consistency across figures; look at the rendered PNG for those.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fig_box = fig.bbox
    problems = []
    if _not_white(fig.get_facecolor()):
        problems.append("figure background is not white")

    text_faults = {}  # one line per kind of fault, with examples, so tick labels do not flood the list
    undrawn = _undrawn_tick_labels(fig)
    for text in fig.findobj(Text):
        s = text.get_text().strip()
        if not s or not text.get_visible() or id(text) in undrawn:
            continue
        label = re.sub(r"\$\\mathdefault\{(.*?)\}\$", r"\1", s).replace("\n", " ")
        label = label if len(label) <= 30 else label[:27] + "..."
        size, colour, face = text.get_fontsize(), text.get_color(), text.get_fontname()
        if size < min_text_pt:
            text_faults.setdefault(f"text below {min_text_pt} pt", []).append(f"{label!r} {size:g} pt")
        if _is_grey(colour):
            text_faults.setdefault("grey text; make it black", []).append(f"{label!r} {colour}")
        if face not in FONT_CHOICES:
            text_faults.setdefault(f"text not in {FONT_CHOICES[0]}", []).append(f"{label!r} {face}")
        box = text.get_window_extent(renderer)
        if box.x0 < fig_box.x0 - 1 or box.x1 > fig_box.x1 + 1 or box.y0 < fig_box.y0 - 1 or box.y1 > fig_box.y1 + 1:
            text_faults.setdefault("text runs off the figure edge", []).append(repr(label))
    for fault, examples in text_faults.items():
        more = f" and {len(examples) - 3} more" if len(examples) > 3 else ""
        problems.append(f"{fault}: {', '.join(examples[:3])}{more}")

    for ax in fig.axes:
        name = ax.get_label() or f"axes {fig.axes.index(ax)}"
        if _not_white(ax.get_facecolor()):
            problems.append(f"{name}: background is not white")
        for side, spine in ax.spines.items():
            if spine.get_visible() and (to_rgb(spine.get_edgecolor()) != (0, 0, 0) or spine.get_linewidth() < FRAME_PT):
                problems.append(f"{name}: {side} spine is not black at {FRAME_PT} pt")
        for line in ax.get_lines():
            if line.get_visible() and line.get_linestyle() not in ("None", "", " ") and line.get_linewidth() < FRAME_PT:
                problems.append(f"{name}: line {line.get_label()!r} is {line.get_linewidth():g} pt, thinner than the "
                                f"{FRAME_PT} pt frame")
        if any(g.get_visible() for g in ax.get_xgridlines() + ax.get_ygridlines()):
            problems.append(f"{name}: grid lines are on (the approved figure has none)")
        if ax.get_legend() is not None:
            problems.append(f"{name}: has a legend; use direct labels next to the data if they fit")
    return problems
