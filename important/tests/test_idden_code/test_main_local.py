"""Focused tests for the student parameter-sweep script."""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from GitHub_QM.important.tests.data_utils import save_array_data, save_csv_table, save_metadata
from GitHub_QM.important.tests.plot_utils import PLOT_STYLE, apply_plot_style
import main_local


def test_safe_float_name(test_subdir):
    assert main_local.safe_float_name(0.001) == "0p0010"
    assert main_local.safe_float_name(2.505) == "2p5050"

    save_metadata(test_subdir / "metadata.json", {"examples": [0.001, 2.505]})


def test_run_one_writes_expected_npz(tmp_path, monkeypatch, test_subdir):
    monkeypatch.setattr(main_local, "OUTDIR", str(tmp_path))

    params = (0, 4, 0.0, 0.0, 0.0, 0.5, 0.0)
    output_path = Path(main_local.run_one(params))
    second_output_path = Path(main_local.run_one(params))

    assert output_path == second_output_path
    assert output_path.exists()
    assert not output_path.with_suffix(".tmp.npz").exists()

    with np.load(output_path, allow_pickle=False) as data:
        keys = set(data.files)
        tlist = data["tlist"]
        rtau_scar = data["Rtau_scar"]
        calculation_time_seconds = float(data["calculation_time_seconds"])
        stored_length = int(data["N"])
        stored_seed = int(data["seed"])

    expected_keys = {
        "seed",
        "tlist",
        "Rtau_scar",
        "N",
        "wd",
        "x",
        "y",
        "z",
        "ds",
        "dd",
        "calculation_time_seconds",
        "t_max",
    }
    assert expected_keys <= keys
    assert stored_length == 4
    assert stored_seed == 0
    assert rtau_scar.shape == tlist.shape
    assert np.all(np.isfinite(rtau_scar))
    assert calculation_time_seconds >= 0.0

    save_array_data(test_subdir / "run_one_output.npz", tlist=tlist, Rtau_scar=rtau_scar)
    save_csv_table(
        test_subdir / "run_one_output.csv",
        {"time": tlist, "Rtau_scar": rtau_scar},
        header="main_local.run_one tiny-output trajectory",
    )

    apply_plot_style()
    fig, ax = plt.subplots(figsize=PLOT_STYLE["figsize"])
    ax.plot(tlist, rtau_scar, marker="o", markersize=3)
    ax.set_xlabel("Time")
    ax.set_ylabel("Rtau_scar")
    ax.set_title("main_local.run_one tiny trajectory")
    ax.grid(True, alpha=PLOT_STYLE["grid_alpha"])
    fig.tight_layout()
    fig.savefig(test_subdir / "run_one_output.png", dpi=PLOT_STYLE["save_dpi"], bbox_inches="tight")
    plt.close(fig)

    save_metadata(
        test_subdir / "metadata.json",
        {
            "output_file": str(output_path.relative_to(tmp_path)),
            "keys": sorted(keys),
            "num_time_points": int(tlist.size),
            "trajectory_npz": "run_one_output.npz",
            "trajectory_csv": "run_one_output.csv",
            "trajectory_plot": "run_one_output.png",
        },
    )
