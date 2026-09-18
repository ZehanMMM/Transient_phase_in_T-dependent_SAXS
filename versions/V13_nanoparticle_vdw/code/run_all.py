"""Run the whole deliverable in order.  python run_all.py [--skip stageA ...]"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

STAGES = [
    ("reference", "digitize_fig_s28e.py",
     "digitise the published Fig. S28E curve"),
    ("validation", "run_validation.py",
     "stage D verification suite (34 checks)"),
    ("stageA", "run_singh_reproduction.py",
     "stage A literature run, ambiguity scan, calibration"),
    ("beta", "run_beta_diagnosis.py",
     "stage A1c why beta = 9.56 nm cannot make a well"),
    ("contact", "run_contact_limit.py",
     "stage A1d the converged potential has no well"),
    ("layout", "run_node_layout_probe.py",
     "stage A2b the 27-element layout ambiguity"),
    ("stageB", "run_convergence.py",
     "stage B continuum convergence at a fixed potential"),
    ("stageC", "run_my_system.py",
     "stage C the present particle system"),
    ("example", "example_usage.py", "minimal API example"),
]

NOTE = """
Stage A takes roughly 20 minutes and stages B and C roughly 25 minutes each on
one core, dominated by the double volume integral.  Run them individually if
you only need one.
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip", nargs="*", default=[])
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()

    print(NOTE.strip())
    print()
    failures = []
    for name, script, description in STAGES:
        if name in args.skip:
            print("SKIP  %s" % name)
            continue
        if args.only and name not in args.only:
            continue
        print("=" * 78)
        print("RUN   %s  (%s)" % (name, description))
        print("=" * 78)
        start = time.time()
        code = subprocess.call([sys.executable, "-u", str(HERE / script)], cwd=HERE)
        print("---   %s finished in %.1f s with exit code %d"
              % (name, time.time() - start, code))
        if code != 0:
            failures.append(name)
    print()
    print("stages run; failures: %s" % (failures or "none"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
