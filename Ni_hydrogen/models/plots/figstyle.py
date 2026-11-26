"""The house plotting style for the figures in models/plots: the research-plots rules (180 x 128 mm,
TeX Gyre Heros or its URW original Nimbus Sans, 17 pt ticks and 20 pt axis titles, a 2.2 pt black frame
with ticks inside on all four sides, curves of 3 pt or more, black text, no grid, no titles), Okabe-Ito
colours with one meaning per kind of series, and every figure written as PDF, 600 dpi PNG and a LaTeX
caption whose numbers come from the results.

Colours (confirmed by the user on 2026-11-24/25):
  routes        geometric black, vibrational vermillion, detailed balance blue (dashed: a diagnostic)
  H loadings    3H sky blue, 5H bluish green, 10H reddish purple
  temperatures  400 K blue, 600 K sky blue, 800 K bluish green, 1000 K orange, 1200 K vermillion
  materials     see MATERIAL_STYLE; NEB paths grey, the lowest-barrier path black

A loading, temperature or material without an entry gets a colour from the spare Okabe-Ito hues, so a
new run never stops the pipeline; add it to the tables to fix its colour.
"""
from __future__ import annotations

import math
import os

import matplotlib as mpl
import numpy as np
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter

from research_style import ANNOT_PT, COLOURS, CURVE_PT, MM, OUTLINE_PT, check_figure, save, use_research_style

KB_EV = 8.617333262e-5
FIGSIZE = (180 * MM, 128 * MM)
FALLBACK_FACE = "Nimbus Sans"      # URW's Helvetica design; TeX Gyre Heros is built from it
_FACE = None


def use_style():
    """The research style, once per process. Where neither TeX Gyre Heros, Helvetica nor Arial is
    installed (Explorer has no TeX), the same design from URW, Nimbus Sans, instead of DejaVu Sans."""
    global _FACE
    if _FACE is not None:
        use_research_style()
        if _FACE == FALLBACK_FACE:
            _fallback_rc(_FACE)
        return _FACE
    face = use_research_style()
    if face == "DejaVu Sans":
        for root in ("/usr/share/fonts", "/usr/local/share/fonts"):
            for dirpath, _dirs, files in os.walk(root):
                for f in files:
                    if f.startswith("NimbusSans-") and f.endswith(".otf"):
                        font_manager.fontManager.addfont(os.path.join(dirpath, f))
        if FALLBACK_FACE in {f.name for f in font_manager.fontManager.ttflist}:
            face = FALLBACK_FACE
            _fallback_rc(face)
    _FACE = face
    return face


def _fallback_rc(face):
    mpl.rcParams.update({"font.sans-serif": [face, "DejaVu Sans"], "mathtext.rm": f"{face}:regular",
                         "mathtext.it": f"{face}:italic", "mathtext.bf": f"{face}:bold"})


def new_figure():
    """A 180 x 128 mm figure and its axes, in the house style."""
    import matplotlib.pyplot as plt
    use_style()
    return plt.subplots(figsize=FIGSIZE)


OKABE = {"black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
         "blue": "#0072B2", "vermilion": "#D55E00", "purple": "#B4508C"}

GREY = "#9A9A9A"            # data that steps back (other NEB paths); never text

ROUTE_STYLE = {   # name: (label, colour, marker, filled, line style)
    "geometric": ("Geometric", OKABE["black"], "o", True, "-"),
    "vibrational": ("Vibrational", OKABE["vermilion"], "s", True, "-"),
    "detailed_balance": ("Detailed balance", OKABE["blue"], "^", False, (0, (5, 3))),
}

MATERIAL_STYLE = {   # stem: (label, colour, marker, line style)
    "Ni_supercell": ("Ni", OKABE["black"], "o", "-"),
    "Al_supercell": ("Al", OKABE["orange"], "s", "-"),
    "Hastelloy_N_7_supercell": ("Hastelloy N 7", OKABE["green"], "D", "-"),
    "Hastelloy_N_42_supercell": ("Hastelloy N 42", OKABE["purple"], "^", (0, (4, 2))),
    # the per-material calculation folders of the cluster project (2026-11-25)
    "316L_SS_bestsqs": ("316L stainless steel", OKABE["black"], "o", "-"),
    "HastelloyN_bestsqs": ("Hastelloy N", OKABE["green"], "D", "-"),
    "HastelloyX_bestsqs": ("Hastelloy X", OKABE["blue"], "s", "-"),
    "Incoloy800_bestsqs": ("Incoloy 800", OKABE["orange"], "^", "-"),
    "Al2O3_oxide": (r"Al$_2$O$_3$", OKABE["sky"], "v", "-"),
    "cr2o3_oxide": (r"Cr$_2$O$_3$", OKABE["vermilion"], "P", "-"),
    "Nickel_oxide": ("NiO", OKABE["purple"], "X", "-"),
}


LOADING_COLOUR = {3: OKABE["sky"], 5: OKABE["green"], 10: OKABE["purple"]}
LOADING_MARKER = {3: "o", 5: "s", 10: "D"}
TEMP_COLOUR = {400.0: OKABE["blue"], 600.0: OKABE["sky"], 800.0: OKABE["green"],
               1000.0: OKABE["orange"], 1200.0: OKABE["vermilion"]}
SPARE = [OKABE["orange"], OKABE["vermilion"], OKABE["blue"], OKABE["black"], OKABE["purple"], OKABE["sky"], OKABE["green"]]
SPARE_MARKERS = ["^", "v", "P", "X", "h", "<", ">"]


def _assign(keys, fixed, spare):
    """{key: value}: the fixed entry where there is one, else the next spare value not already used."""
    used = {fixed[k] for k in keys if k in fixed}
    free = [v for v in spare if v not in used] or list(spare)
    out, i = {}, 0
    for k in keys:
        if k in fixed:
            out[k] = fixed[k]
        else:
            out[k] = free[i % len(free)]
            i += 1
    return out


def loading_styles(n_hs):
    """{n_H: (colour, marker)} for the loadings in one figure."""
    keys = sorted(set(int(n) for n in n_hs))
    c = _assign(keys, LOADING_COLOUR, SPARE)
    m = _assign(keys, LOADING_MARKER, SPARE_MARKERS)
    return {k: (c[k], m[k]) for k in keys}


def temp_styles(temps):
    """{T: colour} for the temperatures in one figure, cold to hot."""
    keys = sorted(set(float(t) for t in temps))
    return _assign(keys, TEMP_COLOUR, SPARE)


def material_styles(stems, pretty=None):
    """{stem: (label, colour, marker, line style)} for the materials in one figure."""
    keys = list(dict.fromkeys(stems))
    fixed_c = {k: MATERIAL_STYLE[k][1] for k in keys if k in MATERIAL_STYLE}
    c = _assign(keys, fixed_c, SPARE)
    m = _assign(keys, {k: MATERIAL_STYLE[k][2] for k in keys if k in MATERIAL_STYLE}, ["o", "s", "D", "^", "v", "P", "X"])
    return {k: (MATERIAL_STYLE[k][0] if k in MATERIAL_STYLE else (pretty(k) if pretty else k), c[k], m[k],
                MATERIAL_STYLE[k][3] if k in MATERIAL_STYLE else "-") for k in keys}


REF_LINE = dict(color="black", lw=2.4, ls=(0, (5, 3)))

def log_ticks(axis, lo, hi):
    """Major ticks on a log axis: plain numbers within 0.001-10,000 over six decades or fewer,
    otherwise powers of ten every 1, 2 or 3 decades."""
    k0, k1 = int(np.ceil(np.log10(lo) - 1e-9)), int(np.floor(np.log10(hi) + 1e-9))
    span = k1 - k0
    if span < 2:
        for subs in ((1, 2, 5), (1, 2, 3, 5, 7), (1, 2, 3, 4, 5, 6, 7, 8, 9)):
            ticks = [m * 10.0 ** k for k in range(k0 - 1, k1 + 1) for m in subs if lo <= m * 10.0 ** k <= hi]
            if len(ticks) >= 2:
                break
        axis.set_major_locator(FixedLocator(ticks))
        axis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10), numticks=200))
        axis.set_minor_formatter(NullFormatter())

        def mantissa(v, _):
            k = int(np.floor(np.log10(v) + 1e-9))
            m = round(v / 10 ** k)
            if -3 <= k <= 3:
                return f"{v:,.{max(0, -k)}f}"
            return rf"$10^{{{k}}}$" if m == 1 else rf"${m}\times10^{{{k}}}$"
        axis.set_major_formatter(FuncFormatter(mantissa))
        return
    step = 1 if span <= 6 else (2 if span <= 12 else 3)
    ks = [k for k in range(k0, k1 + 1) if k % step == 0] or [k0]
    axis.set_major_locator(FixedLocator([10.0 ** k for k in ks]))
    axis.set_minor_locator(LogLocator(base=10, subs=(1.0,), numticks=200) if step > 1
                           else LogLocator(base=10, subs=np.arange(2, 10), numticks=200))
    axis.set_minor_formatter(NullFormatter())
    if span <= 6 and min(ks) >= -3 and max(ks) <= 4:
        axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.{max(0, -int(np.floor(np.log10(v))))}f}"))
    else:
        axis.set_major_formatter(FuncFormatter(lambda v, _: rf"$10^{{{int(round(np.log10(v)))}}}$"))

def set_log_y(ax, values, pad_decades=0.25, frac=0.05):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v) & (v > 0)]
    lo, hi = np.log10(v.min()), np.log10(v.max())
    pad = max(pad_decades, frac * (hi - lo))
    lo, hi = 10 ** (lo - pad), 10 ** (hi + pad)
    ax.set_yscale("log")
    ax.set_ylim(lo, hi)
    log_ticks(ax.yaxis, lo, hi)

def set_linear_y(ax, values, frac=0.07, include=()):
    v = np.concatenate([np.asarray(values, float).ravel(), np.asarray(include, float)])
    v = v[np.isfinite(v)]
    lo, hi = v.min(), v.max()
    pad = frac * ((hi - lo) or abs(hi) or 1.0)
    ax.set_ylim(lo - pad, hi + pad)

def inverse_temperature_axis(ax, temps, min_gap=0.11, pad=0.06):
    """x = 1000/T (K-1), with a top axis in K whose round ticks never crowd and stay within the data."""
    T = np.asarray(sorted(set(float(t) for t in temps)), dtype=float)
    x = 1000.0 / T
    span = x.max() - x.min()
    ax.set_xlim(x.min() - pad * span, x.max() + pad * span)
    ax.set_xlabel(r"1000/$T$ (K$^{-1}$)")
    ax.tick_params(axis="x", top=False)
    top = ax.secondary_xaxis("top", functions=(lambda v: 1000.0 / np.maximum(v, 1e-9),
                                               lambda t: 1000.0 / np.maximum(t, 1e-9)))
    top.set_xlabel("Temperature (K)")
    top.tick_params(axis="x", direction="in", which="both")
    ax._paper_top = (top, T.min(), T.max(), min_gap)
    refresh_ticks(ax)
    return top

def refresh_ticks(ax):
    """Top-axis temperatures and bottom ticks for the final x-limits, never beyond the data."""
    if not hasattr(ax, "_paper_top"):
        return
    top, t_lo, t_hi, min_gap = ax._paper_top
    x0, x1 = ax.get_xlim()
    chosen, last = [], None
    for t in sorted((t for t in range(100, 3001, 100) if t_lo - 1e-6 <= t <= t_hi + 1e-6), reverse=True):
        f = (1000.0 / t - x0) / (x1 - x0)
        if last is None or f - last >= min_gap:
            chosen.append(t)
            last = f
    top.set_xticks(chosen, [str(t) for t in chosen])
    lo, hi = 1000.0 / t_hi, 1000.0 / t_lo
    span = hi - lo
    step = 0.1 if span <= 0.5 else (0.2 if span <= 1.0 else 0.5)
    ticks = [v for v in np.arange(np.floor(lo / step) * step, hi + step, step)
             if lo - 0.06 * span <= v <= hi + 0.06 * span]
    ax.set_xticks(ticks)
    ax.set_xlim(x0, x1)

def temperature_axis(ax, temps):
    temps = sorted(set(float(t) for t in temps))
    pad = 0.06 * (temps[-1] - temps[0])
    ax.set_xlim(temps[0] - pad, temps[-1] + pad)
    ax.set_xticks(temps, [f"{t:.0f}" for t in temps])
    ax.set_xlabel("Temperature (K)")

def text_label(ax, x, y, text, color="black", ha="left", va="center", coords="data", backing=True, **kw):
    return ax.annotate(text, (x, y), xycoords="axes fraction" if coords == "axes" else "data",
                       ha=ha, va=va, fontsize=ANNOT_PT, fontweight="bold", color=color, zorder=7,
                       linespacing=1.15,
                       bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none") if backing else None, **kw)

def _to_axes(ax, x, y):
    return ax.transAxes.inverted().transform(ax.transData.transform(np.column_stack([x, y])))

def end_labels(fig, ax, items, side="right", gap=0.075, max_shift=0.09, room_max=0.42):
    """Name each line beside its end on ``side``: items = [(x_end, y_end, text, colour), ...] in data
    units. The x-axis is widened just enough for the longest name. Labels stack downward where two
    ends nearly coincide; if any label would sit more than ``max_shift`` (axes fraction) away from its
    own line, or the names need more than ``room_max`` of the axis width, nothing is drawn and False
    is returned, so the caller can use a key below the plot instead."""
    renderer = fig.canvas.get_renderer()
    probe = ax.text(0, 0, max((t for _, _, t, _ in items), key=len), fontsize=ANNOT_PT, fontweight="bold",
                    transform=ax.transAxes)
    w_px = probe.get_window_extent(renderer).width + 16
    probe.remove()
    frac = w_px / ax.get_window_extent(renderer).width
    if frac > room_max:
        return False
    x0, x1 = ax.get_xlim()
    grow = frac / (1 - frac)
    if ax.get_xscale() == "log":
        u0, u1 = np.log10(x0), np.log10(x1)
        ax.set_xlim(*((x0, 10 ** (u1 + grow * (u1 - u0))) if side == "right" else (10 ** (u0 - grow * (u1 - u0)), x1)))
    else:
        ax.set_xlim(*((x0, x1 + grow * (x1 - x0)) if side == "right" else (x0 - grow * (x1 - x0), x1)))
    pts = sorted(((tuple(_to_axes(ax, [x], [y])[0]), t, c) for x, y, t, c in items), key=lambda p: -p[0][1])
    placed = []
    for (xf, yf), _, _ in pts:
        placed.append(min(yf, 0.96) if not placed else min(yf, placed[-1] - gap))
    if min(placed) < 0.03 or any(abs(p - pt[0][1]) > max_shift for p, pt in zip(placed, pts)):
        ax.set_xlim(x0, x1)
        return False
    if hasattr(ax, "_paper_top"):
        refresh_ticks(ax)
    elif ax.get_xscale() != "log":                       # no ticks over the room made for the names
        lim = ax.get_xlim()
        ax.set_xticks([t for t in ax.get_xticks() if x0 - 1e-9 <= t <= x1 + 1e-9])
        ax.set_xlim(*lim)
    for ((xf, _), t, c), yl in zip(pts, placed):
        ax.annotate(t, (xf, yl), xycoords="axes fraction", xytext=(8 if side == "right" else -8, 0),
                    textcoords="offset points", ha="left" if side == "right" else "right", va="center",
                    fontsize=ANNOT_PT, fontweight="bold", color=c, zorder=7)
    return True

def key_below(fig, handles, labels, ncol=3, colours=None):
    """A frameless key below the plot; the figure grows by its rows so the plot keeps its size.
    colours: one text colour per entry (the series colour; grey becomes black, as text is never grey)."""
    rows = math.ceil(len(labels) / ncol)
    lines = max(1, max(str(lab).count("\n") + 1 for lab in labels))
    fig.set_figheight(fig.get_figheight() + (8.5 * rows * lines + 2.5 * rows * (lines > 1) + 6) * MM)
    leg = fig.legend(handles, labels, loc="outside lower center", ncol=ncol, frameon=False,
                     handlelength=2.2, handletextpad=0.45, columnspacing=1.1, borderaxespad=0.2,
                     labelspacing=0.55 if lines > 1 else 0.5)
    if colours:
        for text, c in zip(leg.get_texts(), colours):
            text.set_color("black" if c in (GREY, None) else c)
    return leg


def sci(v, sig=3):
    """A number for a key in mathtext: '0.426', '$5.98\\times10^{-7}$', with a true minus sign."""
    v = float(v)
    if v == 0 or not np.isfinite(v):
        return f"{v:g}"
    if 0.01 <= abs(v) < 1000:
        return f"{float(f'{v:.{sig}g}'):g}".replace("-", "−")
    k = int(np.floor(np.log10(abs(v))))
    a_ = float(f"{v / 10 ** k:.{sig}g}")
    sign = "−" if a_ < 0 else ""
    return f"{sign}$10^{{{k}}}$" if abs(a_) == 1 else f"{sign}${abs(a_):g}\\times10^{{{k}}}$"


def energy(v, err=None, digits=3):
    """'0.426 ± 0.020 eV' with a true minus sign."""
    txt = f"{v:.{digits}f}".replace("-", "−")
    if err is not None and np.isfinite(err):
        small = 0 < abs(err) < 0.5 * 10 ** -digits           # would print as 0.000: one significant figure instead
        txt += f" ± {err:.1g}" if small else f" ± {err:.{digits}f}"
    return txt + " eV"


def label_or_key(fig, ax, items, handles, side="right", ncol=3):
    """Direct labels at the line ends when they fit, else a key below. Returns True when keyed."""
    if end_labels(fig, ax, items, side=side):
        return False
    key_below(fig, handles, [t.replace("\n", " ") for _, _, t, _ in items], ncol=ncol)
    return True


# --- labels in free space next to their own lines, with leader lines ---------------------------------

def _frac_box(ax, artist, renderer):
    return artist.get_window_extent(renderer).transformed(ax.transAxes.inverted())


def _fill(pts, step=0.004):
    """Axes-fraction points of a line, filled in every ``step`` between them, inside the axes."""
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) > 1:
        out = [pts[:1]]
        for p, q in zip(pts[:-1], pts[1:]):
            n = max(1, int(np.ceil(np.hypot(*(q - p)) / step)))
            out.append(p + (q - p) * np.linspace(0, 1, n + 1)[1:, None])
        pts = np.vstack(out)
    return pts[(pts[:, 0] > -0.02) & (pts[:, 0] < 1.02) & (pts[:, 1] > -0.02) & (pts[:, 1] < 1.02)]


def _dense_points(ax, xy):
    """Data points of a line -> axes fractions, filled in along the line."""
    with np.errstate(all="ignore"):
        return _fill(ax.transAxes.inverted().transform(ax.transData.transform(np.asarray(xy, float))))


def _obstacles(ax, renderer):
    """Points of every line and mark on the axes (axes fractions), and the boxes of its texts."""
    pts = []
    for ln in ax.get_lines():
        xy = ln.get_xydata()
        if len(xy):
            with np.errstate(all="ignore"):
                p = ax.transAxes.inverted().transform(ln.get_transform().transform(np.asarray(xy, float)))
            drawn = ln.get_linestyle() not in ("None", "none", "") and ln.get_linewidth() > 0
            pts.append(_fill(p) if drawn else p[np.isfinite(p).all(axis=1)])
    from matplotlib.collections import PathCollection
    for c in ax.collections:                              # scatter marks only (not fills or meshes)
        off = c.get_offsets() if isinstance(c, PathCollection) else []
        if len(off):
            with np.errstate(all="ignore"):
                p = ax.transAxes.inverted().transform(c.get_offset_transform().transform(off))
            pts.append(p[np.isfinite(p).all(axis=1)])
    boxes = [_frac_box(ax, t, renderer) for t in ax.texts if t.get_visible() and t.get_text().strip()]
    pts = [p for p in pts if len(p)]
    return (np.vstack(pts) if pts else np.empty((0, 2))), boxes


def line_labels(fig, ax, groups, max_labels=8, margin=0.025, pad=0.012, n=200, step=0.01, keep=400,
                max_tries=40000, leader_from=0.015, max_leader=0.12):
    """Name lines in free space beside them, the way review figures label literature lines.

    groups: [(text, [xy, ...]), ...] - one label per group, xy the data points of each of its lines
    (several lines can share one label, e.g. one source's H and D laws). Each label goes to a spot
    that covers no line, mark or text; the spots are chosen together for the least total distance
    to each label's own lines. A label not right beside a line gets a thin black leader to its
    nearest point. Text is black, never coloured, so colour keeps meaning only the data. Nothing is
    drawn, and False is returned, when there are more than ``max_labels`` groups or a label has no
    free spot within ``max_leader`` (axes fraction) of its line - long leaders across a dense bundle
    read worse than a caption; the caller then names the lines elsewhere (the caption)."""
    if not groups or len(groups) > max_labels:
        return False
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    obst, boxes = _obstacles(ax, renderer)
    occ = np.zeros((n, n))
    if len(obst):
        ij = np.clip((obst * n).astype(int), 0, n - 1)
        occ[ij[:, 1], ij[:, 0]] = 1
    for b in boxes:
        occ[int(np.clip(b.y0 * n, 0, n)):int(np.clip(np.ceil(b.y1 * n), 0, n)),
            int(np.clip(b.x0 * n, 0, n)):int(np.clip(np.ceil(b.x1 * n), 0, n))] = 1
    sat = np.pad(occ.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    texts, homes, spots = [], [], []
    for text, lines in groups:
        t = ax.text(0.5, 0.5, text, transform=ax.transAxes, fontsize=ANNOT_PT, fontweight="bold", color="black",
                    ha="left", va="bottom", zorder=7)
        texts.append(t)
        size = _frac_box(ax, t, renderer)
        w, h = size.width, size.height
        home = [_dense_points(ax, xy) for xy in lines]
        homes.append(home)
        hp = np.vstack([p for p in home if len(p)]) if any(len(p) for p in home) else np.array([[0.5, 0.5]])
        hp = hp[:: max(1, len(hp) // 400)]
        L, B = (a.ravel() for a in np.meshgrid(np.arange(margin, 1 - margin - w, step), np.arange(margin, 1 - margin - h, step)))
        if not len(L):
            spots.append(np.empty((0, 7)))
            continue
        x0, x1, y0, y1 = L - pad, L + w + pad, B - pad, B + h + pad
        i0, i1 = np.clip(np.floor(x0 * n), 0, n).astype(int), np.clip(np.ceil(x1 * n), 0, n).astype(int)
        j0, j1 = np.clip(np.floor(y0 * n), 0, n).astype(int), np.clip(np.ceil(y1 * n), 0, n).astype(int)
        free = (sat[j1, i1] - sat[j0, i1] - sat[j1, i0] + sat[j0, i0]) == 0
        L, B, x0, x1, y0, y1 = (a[free] for a in (L, B, x0, x1, y0, y1))
        dx = np.maximum(np.maximum(x0[:, None] - hp[None, :, 0], hp[None, :, 0] - x1[:, None]), 0)
        dy = np.maximum(np.maximum(y0[:, None] - hp[None, :, 1], hp[None, :, 1] - y1[:, None]), 0)
        cost = np.hypot(dx, dy).min(axis=1) if len(L) else np.empty(0)
        near = cost <= max_leader                            # no long leaders across the plot
        L, B, x0, x1, y0, y1, cost = (a[near] for a in (L, B, x0, x1, y0, y1, cost))
        mine = []
        for k in np.argsort(cost, kind="stable"):
            if any(abs(L[k] - m[0]) < 0.025 and abs(B[k] - m[1]) < 0.025 for m in mine):
                continue
            mine.append((L[k], B[k], x0[k], x1[k], y0[k], y1[k], cost[k]))
            if len(mine) >= keep:
                break
        spots.append(np.array(mine) if mine else np.empty((0, 7)))
    if not all(len(s) for s in spots):
        for t in texts:
            t.remove()
        return False
    best, tries = [np.inf, None], [0]

    def search(remaining, chosen, total):                   # least total distance, no two labels overlapping
        if not remaining:
            best[:] = [total, dict(chosen)]
            return
        floor = {j: spots[j][idx[0], 6] for j, idx in remaining.items()}
        if total + sum(floor.values()) >= best[0] or tries[0] > max_tries:
            return
        i = min(remaining, key=lambda j: len(remaining[j]))
        others = sum(v for j, v in floor.items() if j != i)
        for m in remaining[i]:
            tries[0] += 1
            c = total + spots[i][m, 6]
            if c + others >= best[0] or tries[0] > max_tries:
                break
            bx0, bx1, by0, by1 = spots[i][m, 2:6]
            left, ok = {}, True
            for j, idx in remaining.items():
                if j == i:
                    continue
                o = spots[j][idx]
                kj = idx[(o[:, 3] < bx0) | (o[:, 2] > bx1) | (o[:, 5] < by0) | (o[:, 4] > by1)]
                if not len(kj):
                    ok = False
                    break
                left[j] = kj
            if ok:
                chosen[i] = m
                search(left, chosen, c)
                del chosen[i]

    search({i: np.arange(len(s)) for i, s in enumerate(spots)}, {}, 0.0)
    if best[1] is None:
        for t in texts:
            t.remove()
        return False
    for i, t in enumerate(texts):
        x, y = spots[i][best[1][i], :2]
        t.set_position((x, y))
        box = _frac_box(ax, t, renderer)
        for p in homes[i]:
            if not len(p):
                continue
            ex = np.clip(p[:, 0], box.x0, box.x1)             # nearest point of the label box to the line
            ey = np.clip(p[:, 1], box.y0, box.y1)
            d = np.hypot(p[:, 0] - ex, p[:, 1] - ey)
            k = int(d.argmin())
            if d[k] > leader_from:                     # an annotation, not a data line: thin on purpose
                ax.annotate("", xy=(p[k, 0], p[k, 1]), xytext=(ex[k], ey[k]), xycoords="axes fraction",
                            textcoords="axes fraction", zorder=6,
                            arrowprops=dict(arrowstyle="-", color="black", lw=1.0, shrinkA=1, shrinkB=1))
    return True

def tex_num(v, sig=2):
    """0.66, 18, 490, or $10^{-7}$ and $2.5\\times10^{4}$ outside 0.01-1000; true minus signs."""
    v = float(v)
    if v == 0:
        return "0"
    if 0.01 <= abs(v) < 1000:
        return f"{float(f'{v:.{sig}g}'):g}".replace("-", "$-$")
    k = int(np.floor(np.log10(abs(v))))
    a = float(f"{v / 10 ** k:.{sig}g}")
    body = f"10^{{{k}}}" if abs(a) == 1 else f"{abs(a):g}\\times10^{{{k}}}"
    return f"${'-' if a < 0 else ''}{body}$"

def ev(v, digits=3):
    return f"{v:.{digits}f}".replace("-", "$-$")

def tex_join(items):
    items = list(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]



def points_inside(ax, x, y):
    """The number of finite data points outside the axis limits (0 when all are inside)."""
    (x0, x1), (y0, y1) = sorted(ax.get_xlim()), sorted(ax.get_ylim())
    x, y = np.asarray(x, float).ravel(), np.asarray(y, float).ravel()
    ok = np.isfinite(x) & np.isfinite(y)
    return int((~((x[ok] >= x0) & (x[ok] <= x1) & (y[ok] >= y0) & (y[ok] <= y1))).sum())


def caption_tex(name, module, text):
    return rf"""% Written by models/{module}. Do not edit the numbers by hand.
\begin{{figure}}
  \centering
  \includegraphics[width=180mm]{{{name}.pdf}}
  \caption{{{text}}}
  \label{{fig:{name}}}
\end{{figure}}
"""


def finish(fig, ax, outstem, module, caption, x=None, y=None, keyed=False):
    """Frame on top, the checks, then <outstem>.pdf, <outstem>.png (600 dpi) and <outstem>_caption.tex.
    Problems are printed, never raised: a figure must not stop the pipeline. Returns the PNG path."""
    import matplotlib.pyplot as plt
    for a in fig.axes:
        refresh_ticks(a)
        for spine in a.spines.values():
            spine.set_zorder(10)
    outstem = os.path.splitext(outstem)[0] if outstem.endswith((".png", ".pdf")) else outstem
    name = os.path.basename(outstem)
    problems = [p for p in check_figure(fig) if not (keyed and "legend" in p)]
    if _FACE == FALLBACK_FACE:          # the check knows only TeX Gyre Heros, Helvetica and Arial
        problems = [p for p in problems if not (p.startswith("text not in") and "DejaVu" not in p)]
    if x is not None and points_inside(ax, x, y):
        problems.append(f"{points_inside(ax, x, y)} data points fall outside the axes")
    os.makedirs(os.path.dirname(outstem) or ".", exist_ok=True)
    save(fig, outstem)
    with open(outstem + "_caption.tex", "w") as fh:
        fh.write(caption_tex(name, module, caption))
    plt.close(fig)
    for p in problems:
        print(f"  ! {name}: {p}")
    print(f"  wrote {name}.pdf, .png, _caption.tex")
    return outstem + ".png"


def panel_stem(outfile, suffix):
    """'<dir>/Ni_site_barriers.png' + 'hopa' -> '<dir>/Ni_site_barriers_hopa'."""
    base = os.path.splitext(outfile)[0]
    return f"{base}_{suffix}" if suffix else base
