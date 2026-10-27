"""Compute the Arno habitat time series directly from the supplied data."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Support running this script directly from the tests directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sarawater.habitat import compute_habitat_series

DATA_DIR = Path(__file__).parent / "tests_data"


def main() -> None:
    # Define the species and their corresponding curve codes (in the HQ_curves.txt file) and files containing the habitat time series computed by SimStream.
    species = [
        ("Adult barbel", "BARB_A_R", "Habitat_time_series_Adult_barbel_Arno.txt"),
        ("Juvenile barbel", "BARB_J_R", "habitat_time_series_Juvenile_barbel_Arno.txt"),
    ]

    # Load the flow time series and extract the reference discharge series (Q_series) for the Arno river.
    flow_data = pd.read_csv(
        DATA_DIR / "series_ARNO_2005-2010.txt", sep="\t", parse_dates=[0], index_col=0
    ).apply(pd.to_numeric, errors="coerce")
    flow_series_dates = pd.to_datetime(flow_data.index, format="%d/%m/%Y")
    Q_series = flow_data["REFERENCE"].to_numpy(dtype=float)

    # Load the habitat curves
    habitat_rating_curves = (
        pd.read_csv(DATA_DIR / "HQ_curves.txt", sep="\t")
        .apply(pd.to_numeric, errors="coerce")
        .sort_values("DIS")
    )

    # Load the reference habitat time series computed by SimStream for each species
    reference_habitat_series = {}
    for species_name, curve_code, reference_file in species:
        reference = pd.read_csv(DATA_DIR / reference_file, sep="\t")
        habitat_series_dates = pd.to_datetime(reference["DATE"], format="%d/%m/%Y")
        loaded = pd.to_numeric(reference["HAB_REF [%]"], errors="coerce").to_numpy()
        reference_habitat_series[species_name] = (habitat_series_dates, loaded)

    # Compute the habitat time series for each species using the `compute_habitat_series` function from the `sarawater.habitat` module
    sara_habitat_series = {}
    for species_name, curve_code, reference_file in species:
        habitat_rating_curve = habitat_rating_curves[["DIS", curve_code]].to_numpy(
            dtype=float
        )
        computed = compute_habitat_series(habitat_rating_curve, Q_series)
        sara_habitat_series[species_name] = (flow_series_dates, computed)

    # Plot the discrepancies between the computed and reference habitat time series for each species
    fig, axes = plt.subplots(len(species), 1, sharex=True, figsize=(12, 7))
    for axis, (species_name, curve_code, reference_file) in zip(axes, species):
        habitat_series_dates, loaded = reference_habitat_series[species_name]
        flow_series_dates, computed = sara_habitat_series[species_name]
        if not np.array_equal(habitat_series_dates, flow_series_dates):
            raise ValueError(
                f"Date mismatch for {species_name}: reference dates and computed dates do not match."
            )
        axis.plot(habitat_series_dates, computed - loaded, linewidth=0.8)
        axis.axhline(0, color="black", linewidth=0.7)
        axis.set_title(species_name)
        axis.set_ylabel("Computed − reference (%)")
        axis.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Date")
    fig.suptitle("Habitat time-series discrepancies (REF scenario)")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
