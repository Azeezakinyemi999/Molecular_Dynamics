r"""
report_text.py
==============
Fixed text of the material report (LaTeX snippets, not escaped).

Each section has a 'title', a 'goal', 'docs' (where to read more: (path from the
project root, heading or None, what it explains)), 'columns' (what each table
column means) and 'figures': one entry per kind of figure ('key', used by
report.py) with a 'caption' and 'panels': (panel, what it shows, how to read it,
[(what you see, what it means), ...]).

The wording follows the documentation it points to: the Purpose of each stage in
docs/03-stages/, and the figure catalogue, docs/05-appendices.md section C2.
Every pointer is checked when the report is built and by tests/test_report.py,
so a renamed page or heading is caught rather than left as a dead link.
"""

CONVENTIONS = r"""
\begin{itemize}[leftmargin=*]
\item \textbf{Energies} are in eV. A barrier $E_a$ is the band maximum above the initial state;
  $E_{a,\mathrm{ZPE}}$ adds the zero-point correction from the vibrations. The reverse barrier
  is $E_d$ (desorption for Part 1). $\Delta E$ is the reaction energy, final minus initial state.
\item \textbf{Rates} $k$ and attempt frequencies (prefactors) $\nu$ are in s$^{-1}$. A sound
  prefactor lies roughly within $10^{12}$--$10^{14}$\,s$^{-1}$.
\item \textbf{Loadings.} $n$H is the number of hydrogen atoms in the molecular-dynamics box; at.\,\%
  is the same as a fraction of all atoms. Each loading is its own system: results are never
  averaged over loadings.
\item \textbf{Solubility routes.} \emph{Geometric} and \emph{vibrational} are the two reported
  solubility (and so permeability) routes; they bracket the answer and are never averaged.
  \emph{Detailed balance} is a rate-based cross-check, labelled \emph{diagnostic}: do not quote it.
\item \textbf{Permeability} $\Phi = D\,S$ in mol\,H\,m$^{-1}$\,s$^{-1}$\,Pa$^{-1/2}$, reported as
  $\Phi = \Phi_0 \exp(-E_\Phi/k_BT)$.
\item \textbf{Uncertainties} are one standard deviation. A relative error is a fraction (0.3 = 30\,\%).
  A missing $\pm$ means the quantity carries no uncertainty, which is not the same as a small one.
\item \textbf{``--''} in a table is a value the export does not have.
\end{itemize}
"""

SUMMARY = {
    'title': 'Run summary',
    'docs': [
        ('docs/01-overview.md', None, 'what the workflow computes, and how Parts 1, 2 and 3 fit together'),
        ('docs/03-stages/README.md', None, 'the eleven stages, in the order the sections below follow'),
        ('docs/03-stages/01-materials-and-structures.md', 'Purpose', 'how a material is registered and classed'),
        ('README.md', '5. Getting the results out', r'\texttt{postprocess.py} and \texttt{export\_results.py}, which made the folder this report reads'),
        ('docs/05-appendices.md', 'C. Output files', 'what each pipeline artifact behind the exported tables contains'),
        ('docs/04-practical.md', '1. Troubleshooting', 'when a part is missing or a number looks wrong'),
        ('docs/04-practical.md', '4. Reproducibility checklist', 'what to record to reproduce these numbers'),
    ],
}

SECTIONS = {
    'dissociation': {
        'title': r'Part 1 --- H$_2$ dissociation on the surface',
        'goal': r'Find how H$_2$ breaks into two adsorbed H atoms on the slab surface: the minimum-energy path '
                r'and barrier of every enumerated pathway (nudged elastic band), and which of them have the '
                r'vibrational analysis needed for a trustworthy prefactor.',
        'docs': [
            ('docs/03-stages/05-adsorption-and-enumeration.md', 'Purpose', 'how the surface pathways were placed and chosen'),
            ('docs/03-stages/06-neb.md', 'Purpose', 'the band, its barrier, reaction energy and convergence flag'),
            ('docs/03-stages/06-neb.md', 'Why convergence is recorded rather than enforced', 'why unconverged bands are kept, flagged'),
            ('docs/03-stages/06-neb.md', 'Checks and failure modes', 'what an unconverged band or several maxima mean'),
            ('docs/03-stages/07-vibrations-phva.md', 'Purpose', 'the frequencies behind the zero-point correction and the prefactor'),
            ('docs/03-stages/08-tst-rate-assembly.md', 'Why a failed census does not discard the pathway',
             r'what the \texttt{is\_minimum} and \texttt{ts\_saddle} columns record'),
            ('docs/02-theory.md', '3. Energetics', 'the definitions of the barrier and the reaction energy'),
            ('docs/05-appendices.md', 'Stage 6 — band', 'the catalogue entry of the band figures'),
        ],
        'columns': [
            ('pathway', r'the pathway label: initial site \_\_ the two final sites.'),
            ('IS, FS sites', 'the initial (molecular) site and the two sites the H atoms end on.'),
            (r'$E_a$, $E_{a,\mathrm{ZPE}}$', 'the dissociation barrier from the band, and with the zero-point correction (eV).'),
            (r'$E_\mathrm{des}$', 'the reverse barrier: recombination and desorption from the final state (eV).'),
            (r'$\Delta E$', 'the reaction energy; negative means dissociation is downhill (eV).'),
            ('conv., $f_\\mathrm{max}$', 'whether the band met its force tolerance, and the largest force left on it (eV/\\AA). '
             'An unconverged barrier is provisional.'),
            (r'$\nu_\mathrm{fwd}$, $\nu_\mathrm{rev}$', 'attempt frequencies (s$^{-1}$); empty when the vibrations did not give one.'),
            ('IS min., TS saddle', 'the state-quality verdict of the vibrations: the initial state is a true minimum, and the '
             'transition state has exactly one imaginary mode. Only pathways with both carry a trustworthy prefactor.'),
            ('ZPE source', r'which state the reverse zero-point sum came from: \texttt{FS} (the final state) or a fallback to '
             r'the initial state, in which case the reverse quantities are provisional.'),
        ],
        'figures': [
            {'key': 'mep_overlay_surface', 'caption': 'Every surface dissociation pathway on one axis.',
             'panels': [('Axes', 'Energy relative to the initial state against the reaction coordinate, one line per pathway.',
                         "Each line is one band; its maximum is that pathway's barrier.",
                         [('a wide spread of maxima', 'the sites are not equivalent; the lowest barriers dominate the rate'),
                          ('one line far from the rest', 'an outlier path; look at its own band in Appendix~\\ref{app:pathways}'),
                          ('a line with several maxima', 'more than one elementary step between the endpoints, so the '
                           'pathway is not a single process')])]},
        ],
    },
    'subsurface': {
        'title': 'Part 2 --- Subsurface entry (Hop A and Hop B)',
        'goal': r'Find the barriers for an adsorbed H atom to enter the metal: Hop A from the surface into the first '
                r'subsurface site, and Hop B from there to the second. The first subsurface layer sets the '
                r'solubility; Hop B is computed to show the path continues, not to enter $S$.',
        'docs': [
            ('docs/03-stages/04-subsurface-site-mapping.md', 'Purpose', 'how the subsurface sites and their environments were found'),
            ('docs/03-stages/05-adsorption-and-enumeration.md', 'Why the deeper hop is enumerated but excluded from solubility',
             'why Hop B is computed but not used in $S$'),
            ('docs/03-stages/06-neb.md', 'Checks and failure modes', 'what to do with an unconverged or multi-maximum band'),
            ('docs/03-stages/07-vibrations-phva.md', 'Why a partial Hessian at all', 'the approximation behind every frequency here'),
            ('docs/03-stages/08-tst-rate-assembly.md', 'Why the reverse direction is built from the final state',
             r'what a \texttt{zpe\_source} of FS or a fallback means'),
            ('docs/02-theory.md', '4.2 Zero-point energy', 'the zero-point correction added to the raw barriers'),
            ('docs/05-appendices.md', 'Stage 6 — band', 'the catalogue entry of the full-pathway figure'),
        ],
        'columns': [
            ('hop', 'Hop A: surface H to the first subsurface site; Hop B: first to second subsurface site.'),
            ('pathway, environment', 'the pathway label and the coordination environment of the subsurface site it '
             'enters (e.g.\\ Ni6\\_oct: six Ni neighbours, octahedral).'),
            (r'$E_a$, $E_{a,\mathrm{ZPE}}$', 'the forward barrier, raw from the band and zero-point corrected (eV).'),
            (r'$E_d$, $E_{d,\mathrm{ZPE}}$', 'the reverse barrier, back towards the surface (eV).'),
            (r'$\nu_\mathrm{fwd}$, $\nu_\mathrm{rev}$', 'attempt frequencies (s$^{-1}$).'),
            ('ZPE source', 'as in Part 1: FS, or a fallback whose reverse quantities are provisional.'),
        ],
        'figures': [
            {'key': 'mep_full_pathway', 'caption': 'The whole chain: gas, adsorbed, first and second subsurface.',
             'panels': [('Axes', 'Energy along the full path from H$_2$ in the gas to H in the second subsurface layer, '
                         'one line per pathway.', 'The highest step along a line is the rate-limiting one. This is the '
                         'single most informative figure of the surface half.',
                         [('the largest step is Hop A', 'entry into the metal, not dissociation, limits the surface half'),
                          ('the lines spread widely', 'the pathways differ; the site environment matters')])]},
            {'key': 'mep_overlay_hopa', 'caption': 'Every Hop A pathway on one axis.',
             'panels': [('Axes', 'Energy relative to the adsorbed state against the reaction coordinate, one line per pathway.',
                         'Each maximum is a Hop A barrier.',
                         [('groups of nearly identical lines', 'equivalent sites; the groups are the environments'),
                          ('an outlier', 'check its own band in Appendix~\\ref{app:pathways}')])]},
            {'key': 'mep_overlay_hopb', 'caption': 'Every Hop B pathway on one axis.',
             'panels': [('Axes', 'As for Hop A, from the first to the second subsurface site.', '', [])]},
        ],
    },
    'rates': {
        'title': 'Part 2 --- Rate constants (transition-state theory)',
        'goal': r'Combine each hop barrier with its vibrational frequencies into forward and reverse rate constants '
                r'at every temperature (harmonic transition-state theory), and judge the quality of each state on the way.',
        'docs': [
            ('docs/03-stages/08-tst-rate-assembly.md', 'Purpose', 'how barriers and frequencies become rates'),
            ('docs/03-stages/08-tst-rate-assembly.md', 'Checks and failure modes',
             r'prefactors outside $10^{12}$--$10^{14}$\,s$^{-1}$, fallback zero-point sources'),
            ('docs/02-theory.md', '5.1 Rate', 'the rate expression'),
            ('docs/02-theory.md', '5.2 Vineyard prefactor', 'the attempt frequency'),
            ('docs/05-appendices.md', 'Stage 8 — rates', 'the catalogue entries of the site figures'),
        ],
        'columns': [
            ('site', 'the hop pathway (hopa\\_ or hopb\\_ followed by its site).'),
            (r'$T$', 'temperature (K).'),
            (r'$k_\mathrm{fwd}$, $k_\mathrm{rev}$', 'forward (deeper) and reverse (back out) rate constants (s$^{-1}$).'),
        ],
        'figures': [
            {'key': 'site_barriers', 'caption': 'Barrier per site, by family.',
             'panels': [('Axes', 'One bar or marker per site, grouped by hop family.', '',
                         [('a wide spread within a family', 'the sites are not equivalent')])]},
            {'key': 'site_rates', 'caption': 'The same, as rates at temperature.',
             'panels': [('Axes', 'Rate constant per site.', 'Compare with the barrier figure.',
                         [('the spread grows', 'the exponential amplifies small barrier differences: a few sites carry the flux')])]},
        ],
    },
    'solubility': {
        'title': 'Part 2 --- Solubility',
        'goal': r'Turn the reaction energies into the solubility $S(T)$: how much hydrogen the lattice holds at '
                r'equilibrium per $\sqrt{P}$, from the solution enthalpy of each subsurface environment and its weight. '
                r'An equilibrium quantity: barriers do not enter.',
        'docs': [
            ('docs/03-stages/09-solubility.md', 'Purpose', 'what $S(T)$ is and what it is built from'),
            ('docs/03-stages/09-solubility.md', 'Why two prefactor routes are reported rather than one',
             'geometric and vibrational, and why they differ by orders of magnitude'),
            ('docs/03-stages/09-solubility.md', 'Why the weight is a pathway fraction', 'what the weight column means'),
            ('docs/02-theory.md', '7.1 Per-environment solution enthalpy', 'the per-environment enthalpy'),
            ('docs/02-theory.md', '7.2 Collapsing environments', 'how the environments are combined'),
            ('docs/02b-aggregation.md', '3.1 Two weightings stack, and they are different in kind',
             'why the weight and the Boltzmann factor both act'),
            ('docs/02c-error-propagation.md', r'4. Stage C — per-environment $\Delta H_\text{sol}$', r'where the $\pm$ comes from'),
            ('docs/05-appendices.md', 'Stage 9 — solubility', 'the catalogue entries of the solubility figures'),
        ],
        'columns': [
            ('environment', 'the coordination environment of a first-layer subsurface site.'),
            (r'$\Delta H_\mathrm{sol}$', 'the solution enthalpy of H in that environment, relative to H$_2$ gas (eV).'),
            (r'$\Delta H_\mathrm{hopA}$', 'the entry-hop enthalpy into it (eV).'),
            ('weight', 'the fraction of entry pathways that land in this environment.'),
            ('sites', 'how many sites of this environment were found.'),
        ],
        'figures': [
            {'key': 'env_dH_sol', 'caption': "Solution enthalpy per environment, beside each one's share of $S$.",
             'panels': [('Left and right', 'The enthalpy per environment, and its share of the solubility.', '',
                         [('the environment with the lowest enthalpy is not the largest share',
                           'the pathway weight and the Boltzmann factor pull in different directions; this mismatch is the point of the figure')])]},
            {'key': 'solubility_arrhenius', 'caption': '$S(T)$ per route.',
             'panels': [('Axes', '$\\ln S$ against $1/T$, one line per route.', '',
                         [('a large gap between routes', 'expected: the routes count different things and are never averaged')])]},
        ],
    },
    'diffusivity': {
        'title': 'Part 3 --- Bulk diffusivity',
        'goal': r'Measure how fast dissolved hydrogen moves through the bulk lattice: $D(T)$ from the mean-squared '
                r'displacement in molecular dynamics, reduced to $D_0$ and $E_D$, separately for each loading. '
                r'Independent of the surface: the two chains meet only in the permeability.',
        'docs': [
            ('docs/03-stages/10-bulk-diffusivity.md', 'Purpose', 'what is measured and why it is independent of Part 2'),
            ('docs/03-stages/10-bulk-diffusivity.md', 'Why convergence is judged by halves, not by $R^2$', 'how to judge a $D$'),
            ('docs/03-stages/10-bulk-diffusivity.md', 'Why each loading is kept separate', 'why loadings are not averaged'),
            ('docs/03-stages/10-bulk-diffusivity.md', 'The tension that cannot be designed away',
             'the dilute limit against sampling, behind excluding 1H'),
            ('docs/02-theory.md', '8. Diffusion', 'the MSD, the Einstein relation and the Arrhenius fit'),
            ('docs/02c-error-propagation.md', r'2. Stage A — $\sigma_D$ at one temperature', r'where $\sigma_D$ comes from'),
            ('docs/02c-error-propagation.md', r'3. Stage B — $D(T) \to D_0,\ E_D$', r'how $\sigma$ of $D_0$ and $E_D$ are made'),
            ('docs/05-appendices.md', 'Stage 10 — diffusivity', 'the catalogue entries of the diffusivity figures'),
        ],
        'columns': [
            ('$n$H, at.\\,\\%', 'the loading: H atoms in the box, and as a fraction of all atoms.'),
            (r'$D$', r'the diffusivity at one temperature, $\pm\sigma_D$ (m$^2$/s).'),
            (r'$D_0$, $E_D$', 'the Arrhenius prefactor (m$^2$/s) and activation energy (eV) of the fit over temperature.'),
            (r'$R^2$, points', 'goodness of the fit and how many positive $D$ values entered it; fewer than three '
             'points leave no degrees of freedom, so $R^2$ says nothing.'),
        ],
        'figures': [
            {'key': 'msd_vs_time', 'caption': 'Mean-squared displacement against time, with the fit window shaded, one panel per loading.',
             'panels': [('Each panel', 'The MSD trace of one loading at each temperature.', 'The slope inside the window gives $D$.',
                         [('curvature inside the window', 'not yet diffusive (or a trapping artefact); $D$ is uncertain'),
                          ('a log-log slope away from 1', 'the motion is not ordinary diffusion over this window')])]},
            {'key': 'diffusivity_vs_invT', 'caption': '$D$ against inverse temperature, one panel per loading.',
             'panels': [('Each panel', '$\\ln D$ against $1/T$ for one loading.', '',
                         [('points off the line', 'those temperatures do not follow one activated process')])]},
            {'key': 'arrhenius', 'caption': 'The Arrhenius fit with its parameters, one panel per loading.',
             'panels': [('Each panel', 'The fitted line and its $D_0$, $E_D$.', '',
                         [('fewer than three points', 'no degrees of freedom: the fit is a line through points, not evidence')])]},
            {'key': 'msd_all_concentrations', 'caption': 'Every loading overlaid: mean-squared displacement.',
             'panels': [('Axes', 'MSD traces of all loadings.', '', [])]},
            {'key': 'diffusivity_all_concentrations', 'caption': 'Every loading overlaid: $D(T)$.',
             'panels': [('Axes', '$D$ against $1/T$, one line per loading.', '',
                         [('$D$ not monotonic in loading', 'common; loadings are separate systems, not a series to interpolate')])]},
            {'key': 'arrhenius_params_vs_concentration', 'caption': '$D_0$ and $E_D$ against loading.',
             'panels': [('Axes', 'The two fitted parameters against loading.', '',
                         [('both rising (or falling) together', 'compensation, not faster transport: compare $D$ itself')])]},
            {'key': 'diffusivity_vs_literature', 'caption': 'This work against published laws, each over its own stated range.',
             'panels': [('Axes', '$D$ against $1/T$ with published laws.', '',
                         [('the line outside the published spread', 'a systematic difference of the potential or the setup')])]},
            {'key': 'thermal_expansion', 'caption': 'Lattice parameter against temperature.',
             'panels': [('Axes', 'Lattice parameter from the constant-pressure runs.', '',
                         [('a kink', 'a constant-pressure run did not equilibrate')])]},
        ],
    },
    'permeability': {
        'title': r'Permeability $\Phi = D\,S$',
        'goal': r'Combine the bulk diffusivity (Part 3) with the solubility (Part 2) into the permeability, the '
                r'quantity the workflow exists to produce: $\Phi_0$ and $E_\Phi$ per loading and per solubility route.',
        'docs': [
            ('docs/03-stages/11-permeability-assembly.md', 'Purpose', 'how $\\Phi$ is assembled'),
            ('docs/03-stages/11-permeability-assembly.md', 'Why routes are never merged', 'headline routes and the diagnostic'),
            ('docs/03-stages/11-permeability-assembly.md', 'Why loadings are selected rather than averaged',
             'why each loading has its own row, and why the spread is worth quoting'),
            ('docs/03-stages/11-permeability-assembly.md', 'Checks and failure modes', 'what to check before quoting $\\Phi$'),
            ('docs/02-theory.md', '9.4 Validity', "the dilute-limit assumption behind Sieverts' and Richardson's laws"),
            ('docs/02c-error-propagation.md', r'7. Stage F — $\Phi = D\cdot S$', r'where the $\pm$ of $\Phi_0$ and $E_\Phi$ comes from'),
            ('docs/05-appendices.md', 'Stage 11 — permeability', 'the catalogue entries of the permeability figures'),
        ],
        'columns': [
            ('$n$H', 'the loading whose $D_0$, $E_D$ were used.'),
            ('route, status', 'the solubility route; \\emph{headline} routes are quoted, the \\emph{diagnostic} one is a cross-check.'),
            (r'$\Phi_0$', r'the prefactor (mol\,H\,m$^{-1}$\,s$^{-1}$\,Pa$^{-1/2}$), $\pm$ as a relative error.'),
            (r'$E_\Phi$', r'the activation energy (eV), $\pm\sigma$; $\sigma_{E_\Phi}^2 = \sigma_{E_D}^2 + \sigma_{\Delta H}^2$.'),
            (r'$R^2_S$', 'the goodness of the fit of $S(T)$ for that route.'),
        ],
        'figures': [
            {'key': 'permeability_arrhenius', 'caption': '$\\Phi(T)$ per route with its uncertainty band.',
             'panels': [('Axes', '$\\Phi$ against $1/T$, one line and band per route.', '',
                         [('a wide band', 'dominated by $\\sigma_{E_D}$: the diffusivity fit, not the surface'),
                          ('a large gap between routes', 'expected; quote both as a bracket')])]},
            {'key': 'permeation_summary', 'caption': 'Solubility and permeability side by side, one panel per loading.',
             'panels': [('Each panel', 'The solubility and permeability of one loading.', '',
                         [('the two halves disagree in trend', 'check the inputs of that loading before quoting it')])]},
        ],
    },
}

PATHWAYS = {
    'title': 'Minimum-energy path of every pathway',
    'docs': [
        ('docs/05-appendices.md', 'Stage 6 — band', 'what a single band figure shows'),
        ('docs/03-stages/06-neb.md', 'What a band cannot tell you', 'the limits of one band'),
    ],
    'families': {'dissociation': 'Surface dissociation (Part 1)', 'hopa': 'Hop A (Part 2)', 'hopb': 'Hop B (Part 2)'},
    'verdicts': {'converged': 'converged bands', 'unconverged': 'unconverged bands (barriers provisional)',
                 'unknown': 'bands not in their ranked file (convergence not recorded)'},
}
