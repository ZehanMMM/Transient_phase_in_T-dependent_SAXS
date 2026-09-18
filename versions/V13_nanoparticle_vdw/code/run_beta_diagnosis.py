"""Stage A1c: why beta = 9.56 nm cannot give the published well, quantitatively.

Write U_pair = |U_attr| (x - 1) with x(R) = U_rep / |U_attr|.  Because eps1 and
eps2 are linear prefactors, x is proportional to eps2/eps1 and its SHAPE is set
by beta alone.  Three statements follow, all checked numerically here:

  1. With the literature parameters x(R) peaks at R = 3.00 nm, i.e. beta =
     9.56 nm and eps2/eps1 = 2.231 do place the attraction/repulsion balance
     extremum at the published well position.  That is unlikely to be chance.
  2. The peak value is only 0.455.  Since x < 1 everywhere, U_pair < 0
     everywhere and is strictly monotonic: no well at any separation.
  3. Rescaling eps2 until the peak reaches 1 does not produce the published
     well.  x then crosses 1 twice, so U_pair acquires a positive BARRIER
     between the crossings plus a shallow secondary minimum beyond it, with
     the deep primary minimum still at contact.  A simple -2.33 kcal/mol well
     at 2.99 nm cannot be recovered by any eps2 at fixed beta; beta itself
     must shrink.

Output: outputs/singh_reproduction/A1c_beta_diagnosis.csv
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import npvdw as V

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "singh_reproduction"
EDGE = 13.37


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    params = V.load_parameters("singh_literature")
    gaps = np.concatenate([np.arange(0.10, 8.0, 0.02), np.arange(8.0, 40.01, 0.1)])
    kernels = V.energy_kernels(EDGE, gaps, params)

    attraction, repulsion, total = V.kernel_energies(
        kernels, params.epsilon1, params.epsilon2, params.beta_nm
    )
    ratio = repulsion / np.abs(attraction)
    peak = int(np.argmax(ratio))
    print("1. literature (eps1=%.0f, eps2=%.0f, beta=%.2f nm)"
          % (params.epsilon1, params.epsilon2, params.beta_nm))
    print("   x = U_rep/|U_attr| peaks at R = %.3f nm with x = %.6f"
          % (gaps[peak], ratio[peak]))
    print("   published well position is 2.99 nm, so beta and eps2/eps1 do")
    print("   locate the balance extremum correctly")
    print("2. x < 1 everywhere, so U_pair < 0 everywhere;")
    print("   U_pair strictly increasing in R: %s"
          % bool(np.all(np.diff(total) > 0)))
    print("   the repulsion is short by a factor of %.5f" % (1.0 / ratio[peak]))

    needed = params.epsilon2 / ratio[peak]
    print("3. scaling eps2 to %.3f makes the peak of x exactly 1" % needed)
    rows = []
    for factor, label in ((1.0, "literature"),
                          (1.0 / ratio[peak], "eps2 scaled so max(x) = 1"),
                          (1.5 / ratio[peak], "eps2 scaled so max(x) = 1.5"),
                          (3.0 / ratio[peak], "eps2 scaled so max(x) = 3")):
        eps2 = params.epsilon2 * factor
        a, r, t = V.kernel_energies(kernels, params.epsilon1, eps2, params.beta_nm)
        x = r / np.abs(a)
        crossings = int(np.sum(np.diff(np.sign(x - 1.0)) != 0))
        interior = np.where(
            (t[1:-1] < t[:-2]) & (t[1:-1] < t[2:])
        )[0] + 1
        minima = ["%.3f nm / %+.4f kcal/mol" % (gaps[k], t[k]) for k in interior]
        maxima_idx = np.where((t[1:-1] > t[:-2]) & (t[1:-1] > t[2:]))[0] + 1
        maxima = ["%.3f nm / %+.4f kcal/mol" % (gaps[k], t[k]) for k in maxima_idx]
        print("   eps2 = %9.3f  max(x) = %.4f  x=1 crossings: %d"
              % (eps2, x.max(), crossings))
        print("       interior minima: %s" % (minima or "none"))
        print("       interior maxima: %s" % (maxima or "none"))
        rows.append(
            [
                label,
                "%.4f" % eps2,
                "%.6f" % x.max(),
                "%.3f" % gaps[int(np.argmax(x))],
                crossings,
                "; ".join(minima) or "none",
                "; ".join(maxima) or "none",
            ]
        )

    path = OUT / "A1c_beta_diagnosis.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["case", "epsilon2", "max_U_rep_over_U_attr",
             "gap_at_max_nm", "x_equals_1_crossings", "interior_minima",
             "interior_maxima"]
        )
        writer.writerows(rows)
    print()
    print("Conclusion: no eps2 at beta = 9.56 nm gives a simple well at 2.99 nm.")
    print("wrote %s" % path)


if __name__ == "__main__":
    main()
