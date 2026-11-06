"""
tests/test_report.py
====================
The material report (models/plots/report.py): its documentation pointers resolve,
it builds from an export folder made here, a section appears only when its files
exist, and an export written before manifest.json existed still reports its notes.

The PDF test runs only where latexmk is installed; everything else needs no LaTeX.
"""
import csv
import json
import os
import struct
import sys
import zlib

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOTS = os.path.join(PROJECT_ROOT, 'models', 'plots')
if PLOTS not in sys.path:
    sys.path.insert(0, PLOTS)

import latex_report as L      # noqa: E402
import report                 # noqa: E402
import report_text as RT      # noqa: E402


# ── a made-up export folder ─────────────────────────────────────────────────
def tiny_png(path):
    """A valid 1 x 1 white PNG, without PIL."""
    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data) & 0xffffffff)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as fh:
        fh.write(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', 1, 1, 8, 2, 0, 0, 0))
                 + chunk(b'IDAT', zlib.compress(b'\x00\xff\xff\xff')) + chunk(b'IEND', b''))


def write_csv(path, header, rows):
    with open(path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def part3(d):
    write_csv(os.path.join(d, 'part3_diffusivity_fits.csv'),
              ['n_H', 'at_pct_H', 'D0_m2s', 'D0_err_m2s', 'E_D_eV', 'E_D_err_eV', 'R2_fit', 'n_positive_D_points'],
              [[3, 0.6, 6e-7, 1.6e-7, 0.43, 0.02, 0.994, 5], [5, 1.0, 8e-8, 2.8e-8, 0.28, 0.02, 0.98, 2]])
    write_csv(os.path.join(d, 'part3_diffusivity_vs_T.csv'), ['n_H', 'at_pct_H', 'T_K', 'D_m2s', 'D_err_m2s'],
              [[3, 0.6, 400, 5e-14, 1e-14], [3, 0.6, 800, 1.2e-9, 5e-12], [5, 1.0, 400, 1e-13, '']])
    tiny_png(os.path.join(d, 'figures', '3H', 'msd_vs_time.png'))
    tiny_png(os.path.join(d, 'figures', '5H', 'msd_vs_time.png'))


def full_export(d):
    os.makedirs(d, exist_ok=True)
    write_csv(os.path.join(d, 'part1_dissociation.csv'),
              ['label', 'is_site', 'fs_site1', 'fs_site2', 'Ea_eV', 'E_des_eV', 'delta_E_eV', 'neb_converged',
               'fmax_final', 'Ea_zpe_eV', 'Ed_zpe_eV', 'nu_fwd_s-1', 'nu_rev_s-1', 'is_minimum', 'ts_saddle', 'zpe_source'],
              [['s_1__s_2+s_3', 's_1', 's_2', 's_3', 0.0, 1.1, -1.1, 'True', 0.04, 0.0, 1.1, '', '', 'True', 'False', 'IS_fallback'],
               ['s_4__s_5+s_6', 's_4', 's_5', 's_6', 0.05, 1.2, -1.15, 'False', 0.3, 0.04, 1.18, 3e12, 3.1e12, 'True', 'True', 'FS']])
    write_csv(os.path.join(d, 'part2_hop_barriers.csv'),
              ['hop', 'label', 'env', 'sub1_env', 'sub2_env', 'Ea_raw_eV', 'Ea_zpe_eV', 'Ed_raw_eV', 'Ed_zpe_eV',
               'nu_fwd_s-1', 'nu_rev_s-1', 'zpe_source'],
              [['hopa', 'hopa_s_0', 'Ni4_tet', 'Ni4_tet', '', 0.86, 0.90, 0.10, 0.09, 1.3e12, 3.5e12, 'FS'],
               ['hopb', 'hopb_s_9', 'Ni6_oct', 'Ni6_oct', 'Ni4_tet', 0.20, 0.17, 0.33, 0.30, 5e14, 2e13, 'FS']])
    write_csv(os.path.join(d, 'part2_tst_rates.csv'),
              ['T_K', 'site', 'Ea_zpe_eV', 'Ed_zpe_eV', 'delta_E_eV', 'nu_fwd_s-1', 'nu_rev_s-1', 'k_forward_s-1', 'k_reverse_s-1'],
              [[400, 'hopa_s_0', 0.9, 0.09, 0.77, 1.3e12, 3.5e12, 6.3, 2.9e11], [800, 'hopa_s_0', 0.9, 0.09, 0.77, 1.3e12, 3.5e12, 2e6, 1e12],
               [400, 'hopb_s_9', 0.17, 0.3, -0.1, 5e14, 2e13, 1e12, 3e9], [800, 'hopb_s_9', 0.17, 0.3, -0.1, 5e14, 2e13, 4e13, 1e11]])
    write_csv(os.path.join(d, 'part2_solubility_by_env.csv'),
              ['environment', 'dH_sol_eV', 'dH_sol_err_eV', 'dH_hopA_eV', 'dH_hopA_err_eV', 'weight', 'n_sites'],
              [['Ni6_oct', 0.13, 0.009, 0.66, 0.004, 0.22, 2], ['Ni4_tet', 0.34, 0.019, 0.87, 0.017, 0.78, 7]])
    part3(d)
    write_csv(os.path.join(d, 'overall_permeability.csv'),
              ['n_H', 'route', 'status', 'Phi0_molH_m-1_s-1_Pa-0.5', 'E_phi_eV', 'Phi0_rel_err', 'E_phi_err_eV', 'r2_S'],
              [[3, 'geometric', 'headline', 0.033, 0.578, 0.28, 0.020, 0.995],
               [3, 'vibrational', 'headline', 4.8e-8, 0.537, 0.30, 0.021, 0.979],
               [3, 'detailed_balance', 'diagnostic', 5.2e-7, 0.659, 0.27, 0.019, 0.99999]])
    for name in ('Toy_neb_mep_overlay_surface.png', 'Toy_neb_mep_full_pathway.png', 'Toy_permeability_arrhenius.png'):
        tiny_png(os.path.join(d, 'figures', 'material', name))
    tiny_png(os.path.join(d, 'figures', 'pathways', 'dissociation', 'converged', 's_1__s_2+s_3.png'))
    tiny_png(os.path.join(d, 'figures', 'pathways', 'hopa', 'unconverged', 's_0.png'))
    with open(os.path.join(d, 'manifest.json'), 'w') as fh:
        json.dump({'stem': 'Toy_supercell', 'material': 'Toy', 'source_calculation_dir': '/nowhere/calculation',
                   'exported_at': '2026-11-05T10:00:00', 'loadings': [3, 5], 'single_h_excluded': True,
                   'figures': {}, 'unknown_pathways': {}}, fh)
    with open(os.path.join(d, 'notes.json'), 'w') as fh:
        json.dump(['Part 1: 1 of 2 pathways have both a relaxed IS & a resolved saddle (is_minimum & ts_saddle).',
                   'Overall: `detailed_balance` is a rate-based cross-check.'], fh)
    return d


def tex_of(path):
    with open(path) as fh:
        return fh.read()


# ── documentation pointers ──────────────────────────────────────────────────
def all_pointers():
    for block in (RT.SUMMARY, *RT.SECTIONS.values(), RT.PATHWAYS):
        for path, heading, _ in block['docs']:
            yield block['title'], path, heading


@pytest.mark.parametrize('title,path,heading', list(all_pointers()))
def test_every_documentation_pointer_resolves(title, path, heading):
    assert L.check_pointer(PROJECT_ROOT, path, heading) is None, f'{title}: {path} / {heading}'


def test_every_section_points_somewhere():
    for name, text in RT.SECTIONS.items():
        assert any(p.startswith('docs/03-stages/') for p, _, _ in text['docs']), name


@pytest.mark.parametrize('heading,anchor', [
    ('2.2 Arithmetic mean (A3, A4, A6)', '22-arithmetic-mean-a3-a4-a6'),       # as linked from 11-permeability-assembly.md
    ('Why two prefactor routes are reported rather than one', 'why-two-prefactor-routes-are-reported-rather-than-one'),
    ('Stage 6 — band', 'stage-6--band'),
    ('C2. Figure catalogue', 'c2-figure-catalogue'),                             # as linked from README.md
])
def test_github_slug_matches_the_anchors_the_docs_use(heading, anchor):
    assert L.github_slug(heading) == anchor


# ── LaTeX text ──────────────────────────────────────────────────────────────
def test_tex_helpers():
    assert L.tex_escape('hopa_s_0 & 5%') == r'hopa\_s\_0 \& 5\%'
    assert L.tex_num('0.000123456') == r'$1.23{\times}10^{-4}$'
    assert L.tex_num('') == '--'
    assert L.tex_pm(0.129, 0.0093, lambda v: L.tex_fixed(v, 3)) == r'$0.129 \pm 0.009$'
    assert L.tex_pm(0.129, None, lambda v: L.tex_fixed(v, 3)) == '$0.129$'
    assert L.tex_flag('True') == 'yes' and L.tex_flag('False') == 'no' and L.tex_flag('') == '--'
    assert L.tex_heading(r'Why convergence is judged by halves, not by $R^2$') == \
        r'Why convergence is judged by halves, not by $R^2$'
    assert L.tex_md('use `detailed_balance`') == r'use \texttt{detailed\_balance}'


# ── building ────────────────────────────────────────────────────────────────
def test_full_export_gives_every_section(tmp_path):
    d = full_export(str(tmp_path / 'export' / 'Toy_supercell'))
    path = report.build(d, latex=False, web_base=None, log=lambda *_: None)
    assert path == os.path.join(d, 'report', 'Toy_supercell_report.tex')
    body = tex_of(path)
    for name in report.SECTION_ORDER:
        assert r'\label{sec:' + name + '}' in body, name
        assert r'\section{' + RT.SECTIONS[name]['title'] + '}' in body
    assert body.count(r'\paragraph{Where to read more.}') == len(report.SECTION_ORDER) + 2   # + summary, pathways
    assert r'\texttt{docs/\allowbreak{}03-\allowbreak{}stages/\allowbreak{}06-\allowbreak{}neb.md}' in body
    assert r'\label{app:pathways}' in body and r'\label{app:files}' in body and r'\label{app:docs}' in body
    assert 'Pointers that did not resolve' not in body
    # the caveats, the barrierless check and the prefactor check reach the page
    assert r'is\_minimum \& ts\_saddle' in body
    assert 'no barrier' in body
    assert r'1 of 2 pathways have a prefactor outside' in body
    assert r'\texttt{detailed\_balance} rows are a diagnostic cross-check' in body


def test_sections_appear_only_with_their_files(tmp_path):
    d = str(tmp_path / 'export' / 'Only3_supercell')
    os.makedirs(d)
    part3(d)
    path = report.build(d, latex=False, web_base=None, log=lambda *_: None)
    body = tex_of(path)
    assert r'\label{sec:diffusivity}' in body
    for name in ('dissociation', 'subsurface', 'rates', 'solubility', 'permeability'):
        assert r'\label{sec:' + name + '}' not in body, name
    assert r'\label{app:pathways}' not in body
    assert 'too few for the fit to be evidence' in body        # the 2-point fit of 5H
    assert 'The export recorded no caveats.' in body


def test_summary_md_is_read_when_the_export_predates_the_json(tmp_path):
    d = str(tmp_path / 'export' / 'Old_supercell')
    os.makedirs(d)
    part3(d)
    with open(os.path.join(d, 'SUMMARY.md'), 'w') as fh:
        fh.write('# Old — exported results\n\nSource: `/somewhere/calculation`  ·  stem: `Old_supercell`\n'
                 'Loadings included: 3H, 5H\n\n> Single-H runs are excluded by default.\n\n'
                 '## Read these before quoting any number\n\n1. Part 3: first caveat.\n2. Overall: second caveat.\n')
    rep = report.MaterialReport(d, log=lambda *_: None, web_base=None)
    assert rep.stem == 'Old_supercell'
    assert rep.notes == ['Part 3: first caveat.', 'Overall: second caveat.']
    assert rep.facts['loadings'] == [3, 5] and rep.facts['single_h_excluded']
    body = tex_of(report.build(d, latex=False, web_base=None, log=lambda *_: None))
    assert 'second caveat' in body and 'predates manifest.json' in body


def test_missing_export_folder_is_an_error(tmp_path):
    with pytest.raises(L.ReportError):
        report.build(str(tmp_path / 'nothing'), latex=False, web_base=None, log=lambda *_: None)


def test_export_writes_manifest_and_notes(tmp_path):
    import export_results

    class Run:
        n_H = 3
    export_results.write_manifest('Toy_supercell', str(tmp_path), {'3H': ['a.png']}, {}, ['a note'], [Run()], True)
    facts = json.load(open(tmp_path / 'manifest.json'))
    assert facts['stem'] == 'Toy_supercell' and facts['loadings'] == [3] and facts['single_h_excluded']
    assert json.load(open(tmp_path / 'notes.json')) == ['a note']


@pytest.mark.skipif(L.latexmk_path() is None, reason='LaTeX (latexmk) not installed')
def test_pdf_builds(tmp_path):
    d = full_export(str(tmp_path / 'export' / 'Toy_supercell'))
    pdf = report.build(d, latex=True, web_base=None, log=lambda *_: None)
    assert pdf.endswith('.pdf') and os.path.getsize(pdf) > 10000
