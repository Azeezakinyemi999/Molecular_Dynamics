"""
latex_report.py
===============
The general half of the run report: LaTeX text, bordered tables, figures,
"how to read" blocks, documentation pointers, and the latexmk build. Nothing in
here knows about permeation; report.py holds the sections.

Adapted from the report of Hydrogen-Surface-kinetics (report.py there), so the
reports of the projects look and read the same.
"""
from __future__ import annotations

import math
import os
import re
import shutil
import subprocess

TEXT_WIDTH_CM = 17.4              # A4 with 1.8 cm margins
LATEX_TIMEOUT = 900               # s
TEX_MAX_PT = 16000                # TeX's largest dimension is 16383.99 pt
SCALED_MAX_PX = 6000              # longer side of the copy put in the PDF for such an image


class ReportError(RuntimeError):
    pass


# ── LaTeX text ───────────────────────────────────────────────────────────────
_SPECIAL = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#', '_': r'\_',
            '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}', '^': r'\textasciicircum{}',
            '<': r'\textless{}', '>': r'\textgreater{}', '|': r'\textbar{}'}
_UNICODE = {'−': '$-$', '×': r'$\times$', 'θ': r'$\theta$', 'Φ': r'$\Phi$', '≈': r'$\approx$',
            '→': r'$\rightarrow$', '←': r'$\leftarrow$', '°': r'$^\circ$', 'µ': r'$\mu$', 'μ': r'$\mu$',
            '≤': r'$\le$', '≥': r'$\ge$', '·': r'$\cdot$', '±': r'$\pm$', 'Δ': r'$\Delta$', '…': r'\ldots{}',
            '–': '--', '—': '---', '’': "'", '‘': '`', '“': '``', '”': "''", '²': '$^2$', '³': '$^3$',
            '⁻': '$^-$', '¹': '$^1$', '₀': '$_0$', '₁': '$_1$', '₂': '$_2$', '₃': '$_3$', '₄': '$_4$',
            'σ': r'$\sigma$', 'ν': r'$\nu$', 'Γ': r'$\Gamma$', 'Å': r'\AA{}', '√': r'$\surd$'}
_ASCII = {'−': '-', '×': 'x', 'θ': 'theta', 'Φ': 'Phi', '≈': '~', '→': '->', '←': '<-', '°': 'deg',
          'µ': 'u', 'μ': 'u', '≤': '<=', '≥': '>=', '·': '*', '±': '+/-', 'Δ': 'd', '…': '...', '–': '-',
          '—': '-', '’': "'", '‘': "'", '“': '"', '”': '"', '²': '2', '³': '3', '⁻': '-', '¹': '1',
          '₀': '0', '₁': '1', '₂': '2', '₃': '3', '₄': '4', 'σ': 'sigma', 'ν': 'nu', 'Å': 'A', '►': '>',
          '◄': '<', '─': '-', '═': '=', '│': '|'}


def tex_escape(text):
    """Plain text (labels, file names, notes) -> LaTeX. Unknown non-ASCII -> '?'."""
    out = []
    for ch in str(text):
        if ch in _SPECIAL:
            out.append(_SPECIAL[ch])
        elif ch in _UNICODE:
            out.append(_UNICODE[ch])
        elif ord(ch) < 128:
            out.append(ch)
        else:
            out.append('?')
    return ''.join(out)


def tex_md(text):
    """A note written in Markdown -> LaTeX: `code` becomes typewriter, the rest is escaped."""
    parts = re.split(r'`([^`]*)`', str(text))
    return ''.join(r'\texttt{' + tex_escape(p) + '}' if i % 2 else tex_escape(p) for i, p in enumerate(parts))


def tex_heading(text):
    """A Markdown heading -> LaTeX: its $...$ math kept as it is, `code` in typewriter, the rest escaped."""
    parts = re.split(r'(\$[^$]+\$)', str(text))
    return ''.join(p if i % 2 else tex_md(p) for i, p in enumerate(parts))


def is_missing(v):
    if v is None:
        return True
    if isinstance(v, str):
        return v.strip() == ''
    try:
        return math.isnan(float(v))
    except (TypeError, ValueError):
        return False


def tex_num(v, sig=3):
    """A number in math mode, 3 significant figures; scientific outside [1e-3, 1e5).
    Missing -> '--', booleans -> yes/no, strings escaped."""
    if is_missing(v):
        return '--'
    if isinstance(v, bool) or type(v).__name__ == 'bool_':
        return 'yes' if v else 'no'
    if isinstance(v, str):
        try:
            v = float(v)
        except ValueError:
            return tex_escape(v)
    v = float(v)
    if not math.isfinite(v):
        return r'$\infty$' if v > 0 else r'$-\infty$'
    if v == 0:
        return '$0$'
    a = abs(v)
    if v.is_integer() and a < 1e5:
        return f'${int(v)}$'
    e = math.floor(math.log10(a))
    if -3 <= e < 5:
        if e >= sig - 1:
            s = f'{round(v, sig - 1 - e):.0f}'
        else:
            s = f'{v:#.{sig}g}'.rstrip('.')
            if 'e' in s:
                s = f'{v:.0f}'
        return f'${s}$'
    m = round(v / 10 ** e, sig - 1)
    if abs(m) >= 10:
        m, e = m / 10, e + 1
    return f'${m:.{sig - 1}f}{{\\times}}10^{{{e}}}$'


def tex_fixed(v, digits=3):
    """A number with a fixed count of decimals (energies in eV)."""
    if is_missing(v):
        return '--'
    try:
        return f'${float(v):.{digits}f}$'
    except (TypeError, ValueError):
        return tex_escape(v)


def tex_pm(v, err, fmt=tex_num):
    """'v ± err', the error printed with the same formatter; no error -> 'v'."""
    if is_missing(v):
        return '--'
    s = fmt(v)
    if is_missing(err):
        return s
    return s[:-1] + r' \pm ' + fmt(err)[1:] if s.endswith('$') and fmt(err).startswith('$') else s + r' $\pm$ ' + fmt(err)


def tex_flag(v):
    """True -> yes, False -> no, missing -> --. Accepts the strings a CSV holds."""
    if is_missing(v):
        return '--'
    if isinstance(v, str):
        t = v.strip().lower()
        return 'yes' if t == 'true' else 'no' if t == 'false' else tex_escape(v)
    return 'yes' if bool(v) else 'no'


def tex_path(path):
    """A file path that may break at '/'; escaped typewriter text if \\path cannot take it."""
    path = str(path)
    if re.search(r'[{}%#\\]', path):
        return r'\texttt{' + tex_escape(path) + '}'
    return r'\path{' + path + '}'


def tex_file(name):
    return r'\texttt{' + tex_escape(name) + '}'


def plain_width(tex):
    """Rough printed width (characters) of a LaTeX snippet."""
    s = re.sub(r'\\(textsubscript|mathrm|textbf|emph|texttt|bfseries)\{([^}]*)\}', r'\2', tex)
    s = re.sub(r'\\[a-zA-Z]+', 'x', s)
    s = s.replace(r'\,', ' ')
    s = re.sub(r'[${}^_\\]', '', s)
    return len(s)


def ascii_text(text):
    return ''.join(ch if ord(ch) < 128 else _ASCII.get(ch, '?') for ch in text)


def bullets(items):
    items = [i for i in items if i]
    return (r'\begin{itemize}[leftmargin=*]' + ''.join('\n' + r'\item ' + i for i in items)
            + '\n' + r'\end{itemize}') if items else ''


def enumerated(items):
    items = [i for i in items if i]
    return (r'\begin{enumerate}[leftmargin=*]' + ''.join('\n' + r'\item ' + i for i in items)
            + '\n' + r'\end{enumerate}') if items else ''


def fig_refs(labels):
    """'Figure 3' / 'Figures 3--5, 8' from labels fig:<n>."""
    nums = sorted(int(lab.split(':')[1]) for lab in labels)
    runs = []
    for n in nums:
        if runs and n == runs[-1][1] + 1:
            runs[-1][1] = n
        else:
            runs.append([n, n])
    parts = [rf'\ref{{fig:{a}}}' if a == b else rf'\ref{{fig:{a}}}--\ref{{fig:{b}}}' for a, b in runs]
    return ('Figure ' if len(nums) == 1 else 'Figures ') + ', '.join(parts)


# ── documentation pointers ───────────────────────────────────────────────────
def github_slug(heading):
    """The anchor GitHub gives a Markdown heading: lower case, punctuation dropped,
    spaces to hyphens ('2.2 Arithmetic mean (A3, A4, A6)' -> '22-arithmetic-mean-a3-a4-a6')."""
    s = heading.strip().lower()
    s = re.sub(r'[^\w\- ]', '', s)
    return s.replace(' ', '-')


def doc_headings(path):
    """Every heading of a Markdown file, without its leading #s (code blocks skipped)."""
    out, fenced = [], False
    with open(path, errors='replace') as fh:
        for line in fh:
            if line.lstrip().startswith('```'):
                fenced = not fenced
                continue
            m = None if fenced else re.match(r'#{1,6}\s+(.*?)\s*#*\s*$', line)
            if m:
                out.append(m.group(1))
    return out


def check_pointer(root, path, heading):
    """None if docs/<path> exists and has the heading, else what is wrong."""
    full = os.path.join(root, path)
    if not os.path.isfile(full):
        return f'{path} does not exist'
    if heading and heading not in doc_headings(full):
        return f'{path} has no heading "{heading}"'
    return None


def git_web_base(root):
    """'https://github.com/<owner>/<repo>/blob/<commit>/<prefix>' for the checkout at root,
    or None when it is not a GitHub clone. The commit is HEAD, so links show the docs as
    they were when the report was built."""
    def git(*args):
        try:
            return subprocess.run(['git', '-C', root, *args], capture_output=True, text=True,
                                  timeout=20).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ''
    url, commit, prefix = git('remote', 'get-url', 'origin'), git('rev-parse', 'HEAD'), git('rev-parse', '--show-prefix')
    m = re.match(r'(?:git@github\.com:|https://github\.com/)([^/]+)/(.+?)(?:\.git)?$', url)
    if not (m and commit):
        return None
    return f'https://github.com/{m.group(1)}/{m.group(2)}/blob/{commit}/{prefix}'


# ── the report ───────────────────────────────────────────────────────────────
class LatexReport:
    """LaTeX blocks for one report. out: the folder the report reads (tables, figures);
    dir: the folder the report is written to (out/report)."""

    def __init__(self, out, docs_root=None, web_base=None, max_rows=60, log=print):
        self.out = out
        self.dir = os.path.join(out, 'report')
        self.docs_root, self.web_base = docs_root, web_base
        self.max_rows, self.log = max_rows, log
        self.used = set()                 # output files shown or referenced
        self.n_fig = 0
        self.pointers = []                # (section title, path, heading) of every pointer shown
        self.broken = []                  # pointers whose page or heading was not found

    # tables ------------------------------------------------------------------
    def table(self, rows, columns, caption, source=None, group_by=None, full=False):
        """A bordered longtable. rows: list of dicts. columns: [(key, header LaTeX, align,
        formatter or None)], align 'l', 'r', or 'P<cm>' for a wrapping column. Rows beyond
        max_rows are cut (unless full) and the caption names the file with all of them.
        group_by: a key; a heavier rule separates the rows where its value changes."""
        if not rows:
            return ''
        n_all = len(rows)
        if n_all > self.max_rows and not full:
            rows = rows[:self.max_rows]
        cells = [['--' if is_missing(r.get(k)) else (fmt or tex_num)(r.get(k)) for k, _, _, fmt in columns]
                 for r in rows]
        heads = [h for _, h, _, _ in columns]
        fixed = {}
        for size, (cm_per_char, pad) in ((r'\scriptsize', (0.14, 0.23)), (r'\tiny', (0.115, 0.23))):
            fixed[size] = sum(pad + (0 if a.startswith('P') else cm_per_char * max(
                [max(plain_width(x) for x in h.split(r'\\'))] + [plain_width(c[i]) for c in cells]))
                for i, (_, h, a, _) in enumerate(columns))
        wraps = [float(a[1:]) for _, _, a, _ in columns if a.startswith('P')]
        size, scale, landscape = r'\tiny', 1.0, True
        for sz in (r'\scriptsize', r'\tiny'):
            room = TEXT_WIDTH_CM - 0.4 - fixed[sz]
            if room >= sum(wraps):
                size, landscape = sz, False
                break
            if wraps and room >= 2.0 * len(wraps):
                size, scale, landscape = sz, room / sum(wraps), False
                break
        spec = '|' + '|'.join(r'>{\raggedright\arraybackslash}p{' + f'{float(a[1:]) * scale:.2f}' + 'cm}'
                              if a.startswith('P') else a for _, _, a, _ in columns) + '|'
        head = ' & '.join(r'\makecell{\bfseries ' + h + '}' for h in heads) + r' \\ \thickhline'
        note = ''
        if source:
            self.used.add(source)
            note = (f' First {len(rows)} of {n_all} rows; all of them in {tex_file(source)}.'
                    if len(rows) < n_all else f' File: {tex_file(source)}.')
        lines = [r'\begingroup' + size + r'\setlength{\tabcolsep}{3pt}',
                 r'\begin{longtable}{' + spec + '}',
                 r'\caption{' + caption + note + r'}\\', r'\hline', head, r'\endfirsthead',
                 r'\hline', head, r'\endhead',
                 r'\multicolumn{' + str(len(columns)) + r'}{r}{\emph{continued on the next page}}\\', r'\endfoot',
                 r'\hline', r'\endlastfoot']        # the closing rule stays with the last row on its page
        for k, (r, row) in enumerate(zip(rows, cells)):
            last = k + 1 == len(rows)
            new_group = group_by is not None and not last and rows[k + 1].get(group_by) != r.get(group_by)
            lines.append(' & '.join(row) + (r' \\' if last else r' \\ \thickhline' if new_group else r' \\ \hline'))
        lines += [r'\end{longtable}', r'\endgroup']
        body = '\n'.join(lines)
        return (r'\begin{landscape}' + '\n' + body + '\n' + r'\end{landscape}') if landscape else body

    def pivot(self, rows, index, column, value, caption, index_head, source=None, fmt=None,
              index_fmt=None, col_head=None):
        """One value as a grid: a row per `index` value, a column per `column` value."""
        idx = list(dict.fromkeys(r[index] for r in rows))
        cols = sorted(dict.fromkeys(r[column] for r in rows), key=lambda c: (float(c) if _num(c) else 0, str(c)))
        grid = {(r[index], r[column]): r.get(value) for r in rows}
        recs = [{'_idx': i, **{f'c{j}': grid.get((i, c)) for j, c in enumerate(cols)}} for i in idx]
        spec = [('_idx', index_head, 'l', index_fmt or tex_escape)] + \
               [(f'c{j}', (col_head or tex_escape)(c), 'r', fmt) for j, c in enumerate(cols)]
        return self.table(recs, spec, caption, source)

    @staticmethod
    def column_list(columns):
        return (r'\begin{description}[leftmargin=1.2em,itemsep=0pt,font=\normalfont\itshape]' + '\n'
                + '\n'.join(r'\item[' + c + '] ' + text for c, text in columns) + '\n' + r'\end{description}')

    # figures -----------------------------------------------------------------
    def figure(self, rel, caption, height=0.62):
        """One output figure (path relative to out) as a float. Returns (LaTeX, [label]) or
        ('', []) when the file does not exist."""
        if not rel or not os.path.isfile(os.path.join(self.out, rel)):
            return '', []
        self.used.add(rel)
        self.n_fig += 1
        label = f'fig:{self.n_fig}'
        return (r'\begin{figure}[H]\centering'
                rf'\includegraphics[width=\linewidth,height={height}\textheight,keepaspectratio]{{{self.image(rel)}}}'
                r'\caption{' + caption + r' File: ' + tex_file(rel) + r'.}\label{' + label + r'}\end{figure}'), [label]

    def figure_grid(self, rels, captions, caption, per_row=3):
        """Small images in a grid, each with its own short caption, under one caption.
        Returns (LaTeX, [label])."""
        rels = [r for r in rels if os.path.isfile(os.path.join(self.out, r))]
        if not rels:
            return '', []
        self.n_fig += 1
        label = f'fig:{self.n_fig}'
        width = 0.98 / per_row
        cells = []
        for k, (rel, cap) in enumerate(zip(rels, captions)):
            self.used.add(rel)
            cells.append(r'\begin{minipage}[t]{' + f'{width:.3f}' + r'\linewidth}\centering'
                         rf'\includegraphics[width=\linewidth,height=0.2\textheight,keepaspectratio]{{{self.image(rel)}}}'
                         r'\\[-2pt]{\scriptsize ' + cap + r'}\end{minipage}' + ('\\par\\medskip' if (k + 1) % per_row == 0 else '\\hfill'))
        return (r'\begin{figure}[H]\centering' + '\n' + '\n'.join(cells) + '\n'
                + r'\caption{' + caption + r'}\label{' + label + r'}\end{figure}'), [label]

    def image(self, rel):
        """The path, from the report folder, of the image to include for output file rel. A PNG
        too large for TeX at its own resolution goes in as a smaller copy in report/scaled/."""
        if not rel.lower().endswith('.png'):
            return '../' + rel
        try:
            from PIL import Image
        except ImportError:
            return '../' + rel
        limit, Image.MAX_IMAGE_PIXELS = Image.MAX_IMAGE_PIXELS, None
        try:
            with Image.open(os.path.join(self.out, rel)) as im:
                w, h = im.size
                dpi = (im.info.get('dpi') or (72, 72))[0] or 72
                if max(w, h) * 72 / dpi <= TEX_MAX_PT:
                    return '../' + rel
                scale = SCALED_MAX_PX / max(w, h)
                small = im.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
        except OSError:
            return '../' + rel
        finally:
            Image.MAX_IMAGE_PIXELS = limit
        dest = 'scaled/' + rel.replace('/', '__')
        os.makedirs(os.path.join(self.dir, 'scaled'), exist_ok=True)
        small.save(os.path.join(self.dir, dest))
        self.log(f'  report: {rel} is {w} x {h} px, too large for LaTeX; included scaled ({dest})')
        return dest

    @staticmethod
    def how_to_read(spec, labels):
        """Per figure: what it shows, how to read it, and what each thing you may see means."""
        out = [r'\subsubsection*{' + fig_refs(labels) + ': ' + spec['caption'] + '}',
               r'\begin{description}[leftmargin=1.2em,style=nextline]']
        for name, shows, read, trends in spec['panels']:
            out.append(r'\item[' + name + '] ' + shows + ' ' + read)
            if trends:
                out.append(r'\begin{itemize}[leftmargin=1.4em,itemsep=0pt,topsep=2pt]')
                out += [r'\item \emph{' + see + r'} $\Rightarrow$ ' + means + '.' for see, means in trends]
                out.append(r'\end{itemize}')
        out.append(r'\end{description}')
        return '\n'.join(out)

    # documentation pointers --------------------------------------------------
    def pointer_link(self, path, heading=None):
        """'docs/03-stages/06-neb.md, section "Purpose"', linked to the page on GitHub when
        the checkout has a GitHub remote. A page or heading that is not there is logged."""
        problem = check_pointer(self.docs_root, path, heading) if self.docs_root else None
        if problem:
            self.broken.append(problem)
            self.log(f'  report: documentation pointer: {problem}')
        text = r'\texttt{' + tex_escape(path).replace('/', r'/\allowbreak{}').replace('-', r'-\allowbreak{}') + '}'
        if heading:
            text += r', \S\,``' + tex_heading(heading) + "''"
        if self.web_base and not problem:
            url = self.web_base + path + ('#' + github_slug(heading) if heading else '')
            text = r'\href{' + url.replace('%', r'\%').replace('#', r'\#') + '}{' + text + '}'
        return text

    def pointers_block(self, title, docs):
        """'Where to read more': one line per (path, heading or None, what it explains)."""
        if not docs:
            return ''
        items = []
        for path, heading, why in docs:
            self.pointers.append((title, path, heading))
            items.append(self.pointer_link(path, heading) + (' --- ' + why if why else ''))
        return (r'\begin{itemize}[leftmargin=*,itemsep=1pt,topsep=2pt]' + ''.join('\n' + r'\item ' + i for i in items)
                + '\n' + r'\end{itemize}')

    # sections ----------------------------------------------------------------
    def section(self, name, text, parts):
        """One section: goal, where to read more, then parts, then how to read the tables and
        each figure shown. text: the section's entry of report_text.SECTIONS. parts:
        (subsection title or None, LaTeX) or ('figure', key, file[, caption addition]) or
        ('grid', key, files, captions[, images per row])."""
        specs = {f['key']: f for f in text.get('figures', [])}
        body = [r'\section{' + text['title'] + '}', r'\label{sec:' + name + '}',
                r'\paragraph{Goal.} ' + text['goal']]
        if text.get('docs'):
            body += [r'\paragraph{Where to read more.}', self.pointers_block(text['title'], text['docs'])]
        shown = {}
        for part in parts:
            if part[0] in ('figure', 'grid'):
                key = part[1]
                if part[0] == 'figure':
                    extra = part[3] if len(part) > 3 else ''
                    tex, labels = self.figure(part[2], specs[key]['caption'] + (' ' + extra if extra else ''))
                else:
                    tex, labels = self.figure_grid(part[2], part[3], specs[key]['caption'],
                                                   part[4] if len(part) > 4 else 3)
                if labels:
                    body.append(tex)
                    shown.setdefault(key, []).extend(labels)
                continue
            title, content = part
            if not content:
                continue
            if title and content.startswith(r'\begin{landscape}'):
                content = content.replace(r'\begin{landscape}', r'\begin{landscape}' + '\n' + r'\subsection*{' + title + '}', 1)
            elif title:
                body.append(r'\subsection*{' + title + '}')
            body.append(content)
        if text.get('columns'):
            body += [r'\subsection*{How to read the tables}', self.column_list(text['columns'])]
        if shown:
            body.append(r'\subsection*{How to read the figures}')
            body += [self.how_to_read(specs[key], labels) for key, labels in shown.items()]
        return '\n\n'.join(body)


def _num(x):
    try:
        float(x)
        return True
    except (TypeError, ValueError):
        return False


# ── document ─────────────────────────────────────────────────────────────────
def preamble(title, macros=()):
    return '\n'.join([
        r'\documentclass[10pt,a4paper]{article}',
        r'\usepackage[T1]{fontenc}', r'\usepackage[utf8]{inputenc}', r'\usepackage{lmodern}',
        r'\usepackage[margin=1.8cm]{geometry}', r'\usepackage{amsmath,amssymb}', r'\usepackage{graphicx}',
        r'\usepackage{longtable,array,makecell}', r'\usepackage{xcolor}', r'\usepackage{float}',
        r'\usepackage{caption}', r'\usepackage{enumitem}', r'\usepackage{pdflscape}', r'\usepackage{listings}',
        r'\usepackage[colorlinks=true,linkcolor=black,citecolor=black,urlcolor=blue!55!black]{hyperref}',
        r'\usepackage{url}',
        r'\expandafter\def\expandafter\UrlBreaks\expandafter{\UrlBreaks\do\-\do\_}',
        r'\newcommand{\thickhline}{\noalign{\global\arrayrulewidth=0.9pt}\hline\noalign{\global\arrayrulewidth=0.4pt}}',
        r'\renewcommand{\arraystretch}{1.18}', r'\setlength{\parindent}{0pt}', r'\setlength{\parskip}{4pt}',
        r'\setlength{\LTpre}{6pt}', r'\setlength{\LTpost}{8pt}',
        r'\captionsetup{font=small,labelfont=bf,justification=raggedright,singlelinecheck=true}',
        r'\captionsetup[longtable]{font=small,labelfont=bf}',
        r'\lstset{basicstyle=\ttfamily\scriptsize,breaklines=true,columns=fullflexible,frame=single,'
        r'commentstyle=\color{gray},keywordstyle=\bfseries,language=Python}',
        *macros,
        r'\hypersetup{pdftitle={' + re.sub(r'[\\${}^_]', '', title) + '}}',
    ])


def title_page(title, subtitle, facts, footer):
    """facts: [(name, LaTeX value)]."""
    rows = '\n'.join(r'\textbf{' + k + '} & ' + v + r' \\ \hline' for k, v in facts)
    return '\n'.join([
        r'\begin{titlepage}\centering', r'\vspace*{3cm}',
        r'{\LARGE\bfseries ' + title + r'\par}', r'\vspace{0.6cm}',
        r'{\large ' + subtitle + r'\par}', r'\vspace{1.5cm}',
        r'\small\renewcommand{\arraystretch}{1.4}',
        r'\begin{tabular}{|>{\raggedright\arraybackslash}p{4cm}|>{\raggedright\arraybackslash}p{11.5cm}|}\hline',
        rows, r'\end{tabular}', r'\vfill', r'{\small ' + footer + r'\par}', r'\end{titlepage}'])


def latexmk_path(explicit=None):
    for p in (explicit, shutil.which('latexmk'), '/Library/TeX/texbin/latexmk'):
        if p and os.path.exists(p):
            return p
    return None


def latex_errors(log_path, n=12):
    if not os.path.exists(log_path):
        return 'no LaTeX log'
    with open(log_path, errors='replace') as f:
        lines = f.read().splitlines()
    hits = [i for i, line in enumerate(lines) if line.startswith('!') or ':' in line[:200] and 'Error' in line]
    return '\n'.join('\n'.join(lines[i:i + 3]) for i in hits[:n]) or '\n'.join(lines[-n:])


def build_pdf(tex, exe, log=print):
    """latexmk -pdf on the .tex file; returns the PDF path. Raises ReportError on failure."""
    folder, stem = os.path.dirname(tex), os.path.splitext(os.path.basename(tex))[0]
    env = dict(os.environ, PATH=os.path.dirname(exe) + os.pathsep + os.environ.get('PATH', ''))
    cmd = [exe, '-pdf', '-interaction=nonstopmode', '-halt-on-error', '-file-line-error', os.path.basename(tex)]
    log(f'report: building {stem}.pdf with LaTeX ...')
    try:
        proc = subprocess.run(cmd, cwd=folder, capture_output=True, text=True, errors='replace',
                              timeout=LATEX_TIMEOUT, env=env)
    except subprocess.TimeoutExpired:
        raise ReportError(f'LaTeX took longer than {LATEX_TIMEOUT} s on {tex}') from None
    pdf = os.path.join(folder, stem + '.pdf')
    if proc.returncode != 0 or not os.path.exists(pdf):
        raise ReportError(f'LaTeX failed on {tex}:\n' + latex_errors(os.path.join(folder, stem + '.log')))
    subprocess.run([exe, '-c', os.path.basename(tex)], cwd=folder, capture_output=True, env=env)
    return pdf
