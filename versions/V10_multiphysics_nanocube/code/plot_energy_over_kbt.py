"""Plot the V10.18 Brownian-rotation energies in units of kBT.

The underlying 10 K calculations are read without modification from the
V10.18 CSV.  Only the energy normalization and display interpolation differ,
so the signed-energy crossings are identical to the joule-valued version.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from scipy.interpolate import PchipInterpolator


plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "axes.linewidth": 1.8,
        "axes.grid": False,
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = ROOT / "versions" / "V10_multiphysics_nanocube" / "outputs"
SOURCE_CSV = OUTPUT_DIR / "brownian_cube_rotation_pair_energies.csv"
EXPERIMENTAL_MIN_K = 233.15
EXPERIMENTAL_MAX_K = 293.15
TRANSIENT_MIN_K = 253.15
TRANSIENT_MAX_K = 273.15
EXPERIMENTAL_WINDOW_S = 20.0


def main():
    if not SOURCE_CSV.exists():
        raise FileNotFoundError(
            f"Run v10_18_brownian_cube_rotation.py first: {SOURCE_CSV}"
        )
    data = np.genfromtxt(SOURCE_CSV, delimiter=",", names=True)
    temperature_K = np.asarray(data["temperature_K"])
    kBT_J = np.asarray(data["kBT_J"])
    order = np.argsort(temperature_K)
    display_temperature_K = np.linspace(200.0, 300.0, 501)

    def normalized(field: str) -> np.ndarray:
        return np.asarray(data[field]) / kBT_J

    def smooth(values: np.ndarray) -> np.ndarray:
        return PchipInterpolator(
            temperature_K[order], values[order]
        )(display_temperature_K)

    face_equilibrium = normalized("face_equilibrium_Edd_J")
    face_neel_penalty = normalized("face_Neel_penalty_J")
    face_brownian_penalty = normalized(
        "face_Brownian_constraint_penalty_J"
    )
    face_magnetic = normalized("face_constrained_Emag_J")
    tip_neel_only = normalized("tip_Neel_only_Emag_J")
    tip_magnetic = normalized("tip_Neel_plus_Brownian_Emag_J")
    face_vdw = normalized("face_vdW_pair_J")
    tip_vdw = normalized("tip_vdW_pair_J")
    face_pair_total = normalized("face_pair_total_J")
    tip_pair_total = normalized("tip_pair_total_J")

    figure, ax = plt.subplots(figsize=(9.5, 9.5))
    figure.subplots_adjust(left=0.14, right=0.97, bottom=0.18, top=0.85)
    ax.axvspan(
        EXPERIMENTAL_MIN_K,
        EXPERIMENTAL_MAX_K,
        color="#c9c9c9",
        alpha=0.22,
        zorder=0,
    )
    ax.axvspan(
        TRANSIENT_MIN_K,
        TRANSIENT_MAX_K,
        color="#f6e58d",
        alpha=0.48,
        zorder=0,
    )
    ax.plot(
        display_temperature_K,
        smooth(face_magnetic),
        color="#b45309",
        linewidth=2.8,
        label=r"$E_{mag}^{face}$",
    )
    ax.plot(
        display_temperature_K,
        smooth(tip_magnetic),
        color="#e08a1e",
        linewidth=2.8,
        label=r"$E_{mag}^{tip}$",
    )
    ax.plot(
        display_temperature_K,
        smooth(face_vdw),
        color="#2676b8",
        linewidth=2.8,
        label=r"$E_{vdW}^{face,pair}$",
    )
    ax.plot(
        display_temperature_K,
        np.ones_like(display_temperature_K),
        color="#222222",
        linewidth=2.8,
        label=r"$k_BT=1$",
    )
    ax.axhline(0.0, color="#777777", linewidth=0.9)
    ax.set_xlim(200.0, 300.0)
    normalized_plot_values = np.concatenate(
        [
            smooth(face_magnetic),
            smooth(tip_magnetic),
            smooth(face_vdw),
            np.ones_like(display_temperature_K),
            np.array([0.0]),
        ]
    )
    normalized_max = float(np.max(normalized_plot_values))
    ax.set_ylim(
        -12.0,
        normalized_max + 0.08 * max(normalized_max + 12.0, 1.0),
    )
    ax.set_xticks(np.arange(200.0, 301.0, 10.0))
    ax.set_xlabel("Temperature (K)", fontsize=21, fontweight="bold")
    ax.set_ylabel(r"Energy / $k_BT$", fontsize=21, fontweight="bold")
    ax.tick_params(
        axis="both",
        which="both",
        direction="in",
        top=True,
        right=True,
        width=1.6,
        length=6,
        labelsize=18,
    )
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_linewidth(1.8)
    ax.grid(False, which="both", axis="both")
    ax.xaxis.grid(False, which="both")
    ax.yaxis.grid(False, which="both")
    for gridline in ax.get_xgridlines() + ax.get_ygridlines():
        gridline.set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    handles.extend(
        [
            Patch(
                facecolor="#c9c9c9",
                alpha=0.22,
                edgecolor="none",
                label="experimental range",
            ),
            Patch(
                facecolor="#f6e58d",
                alpha=0.48,
                edgecolor="none",
                label="transient aggregation",
            ),
        ]
    )
    labels.extend(["experimental range", "transient aggregation"])
    ax.legend(
        handles,
        labels,
        loc="lower left",
        bbox_to_anchor=(0.015, 0.015),
        ncol=1,
        frameon=True,
        facecolor="white",
        edgecolor="black",
        framealpha=1.0,
        prop={"family": "Arial", "size": 14.0, "weight": "bold"},
        handlelength=2.8,
        handletextpad=0.9,
        borderpad=0.85,
        labelspacing=0.65,
        borderaxespad=0.0,
    )
    figure.suptitle(
        r"Pair energies normalized by $k_BT$"
        f"\n16 nm Fe$_3$O$_4$, 1.5 nm ligand shell; {EXPERIMENTAL_WINDOW_S:g} s window",
        fontsize=22,
        fontweight="bold",
    )
    figure.text(
        0.42,
        0.027,
        r"Signed energies are divided by the local $k_BT$; the underlying V10.18 calculations are unchanged."
        "\n"
        r"The face curve includes the Néel relaxation deficit and suppressed-rotation penalty.",
        ha="center",
        va="bottom",
        fontsize=10.5,
        fontweight="bold",
        color="#555555",
    )

    png_path = OUTPUT_DIR / "brownian_cube_rotation_E_over_kBT.png"
    pdf_path = OUTPUT_DIR / "brownian_cube_rotation_E_over_kBT.pdf"
    csv_path = OUTPUT_DIR / "brownian_cube_rotation_E_over_kBT.csv"
    figure.savefig(png_path, dpi=600)
    figure.savefig(pdf_path)
    plt.close(figure)

    np.savetxt(
        csv_path,
        np.column_stack(
            [
                temperature_K,
                face_equilibrium,
                face_neel_penalty,
                face_brownian_penalty,
                face_magnetic,
                tip_neel_only,
                tip_magnetic,
                face_vdw,
                tip_vdw,
                face_pair_total,
                tip_pair_total,
                np.ones_like(temperature_K),
            ]
        ),
        delimiter=",",
        header=(
            "temperature_K,face_equilibrium_Edd_over_kBT,"
            "face_Neel_penalty_over_kBT,"
            "face_Brownian_constraint_penalty_over_kBT,"
            "face_constrained_Emag_over_kBT,"
            "tip_Neel_only_Emag_over_kBT,"
            "tip_Neel_plus_Brownian_Emag_over_kBT,"
            "face_vdW_pair_over_kBT,tip_vdW_pair_over_kBT,"
            "face_pair_total_over_kBT,tip_pair_total_over_kBT,"
            "kBT_over_kBT"
        ),
        comments="",
        fmt="%.12e",
    )
    print(png_path)
    print(pdf_path)
    print(csv_path)


if __name__ == "__main__":
    main()
