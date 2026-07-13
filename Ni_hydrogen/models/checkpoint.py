"""
models/checkpoint.py
=====================
Uniform ``.done`` sentinel markers for Section/Phase/Step resume logic
across all workflow files (neb_workflow, diffusivity_workflow,
permeation_workflow, vibrations, neb_subsurface).

Convention: one ``.done`` file per Step, named after the step, colocated
in that step's own output directory (e.g. ``{job_dir}/fsmin.done`` next
to ``neb_final_relaxed.lammps``). ``mark_done()`` is always the last call
in a Step, only after its real output has been confirmed to exist --
matches the existing rationale already used for NEB's FS-min skip check
(the real output is the last thing written, so a killed job never leaves
a partial output file that could be mistaken for done).

Usage
-----
    from models.checkpoint import is_done, mark_done

    marker = os.path.join(job_dir, 'fsmin.done')
    if is_done(marker):
        print(f'  [{label}] already done -- skipping')
    else:
        ...  # do the work
        mark_done(marker)
"""
from pathlib import Path


def is_done(marker_path) -> bool:
    return Path(marker_path).exists()


def mark_done(marker_path) -> None:
    p = Path(marker_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.touch()
