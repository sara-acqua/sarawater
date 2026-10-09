"""
This module provides plotting functionality for comparing scenarios in a reach.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.axes import Axes
from matplotlib.ticker import MaxNLocator
from typing import List, Optional, Union
from datetime import datetime

from sarawater.reach import Reach
from sarawater.utils import _compute_date_mask


class ReachPlotter:
    """A class for plotting and comparing scenarios in a reach."""

    def __init__(
        self,
        reach: Reach,
        output_dir: Optional[str] = "outputs",
        scenario_colors: List[str] | None = None,
    ):
        """
        Initialize a ReachPlotter instance.

        Parameters
        ----------
        reach : Reach
            The reach object containing scenarios to plot
        output_dir : str, optional
            Directory where to save the plots. By default "outputs". The directory is created if it does not exist. Note that plotting methods need to be called with save=True to save the plots.
        scenario_colors : list of str, optional
            List of colors to use for each scenario in the plots. Default is a set of distinct tab colors.
        """
        self.reach = reach
        if output_dir is None:
            raise ValueError(
                "output_dir must be a valid directory path string and cannot be None"
            )
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        if scenario_colors is not None:
            self.scenario_colors = list(scenario_colors)[: len(self.reach.scenarios)]
        else:
            self.scenario_colors = [
                "tab:orange",
                "tab:green",
                "tab:red",
                "tab:purple",
                "tab:brown",
                "tab:pink",
                "tab:gray",
                "tab:olive",
                "tab:cyan",
            ][: len(self.reach.scenarios)]

    def _ensure_iha_dir(self) -> str:
        """Create IHA subfolder if it doesn't exist (for multi-file methods)."""
        iha_dir = os.path.join(self.output_dir, "IHA_plots")
        os.makedirs(iha_dir, exist_ok=True)
        return iha_dir

    def plot_scenarios_discharge(
        self,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        log_scale: bool = True,
        save: bool = False,
        plot_Qnat: bool = True,
    ) -> Axes:
        """
        Plot discharge comparison between scenarios.

        Parameters
        ----------
        start_date : str or datetime, optional
            Start date for the plot range
        end_date : str or datetime, optional
            End date for the plot range
        log_scale : bool, default=True
            Whether to use log scale for y-axis
        save : bool, default=False
            Whether to save the plot to file
        plot_Qnat : bool, default=True
            Whether to plot the natural flow (Qnat)
        """
        mask = _compute_date_mask(self.reach.dates, start_date, end_date)

        plt.figure()
        for color, scenario in zip(self.scenario_colors, self.reach.scenarios):
            scenario_Qrel = scenario._require_Qrel()
            plt.plot(
                np.array(self.reach.dates)[mask],
                scenario_Qrel[mask],
                color=color,
                label=scenario.name,
            )

        if plot_Qnat:
            plt.plot(
                np.array(self.reach.dates)[mask],
                self.reach.Qnat[mask],
                color="tab:blue",
                label="Natural",
            )

        if log_scale:
            plt.yscale("log")
        plt.grid(True)
        plt.legend()
        plt.title(f"{self.reach.name} - Discharge Comparison")
        plt.xlabel("Date")
        plt.ylabel("Discharge (m³/s)")

        if save:
            plt.savefig(
                os.path.join(self.output_dir, "discharge_comparison.png"),
                bbox_inches="tight",
            )
        return plt.gca()

    def plot_iari_groups(
        self,
        save: bool = False,
        ylims: Optional[List[Optional[tuple[float, float]]]] = None,
    ) -> Axes:
        """
        Plot IARI values comparison for each group.

        Parameters
        ----------
        save : bool, default=False
            Whether to save the plots to files
        ylims : list[tuple[float, float] | None] | None, default=None
            Y-axis limit for each group. If None, the limit is set automatically.
        """
        if ylims is None:
            ylims = [None, None, None, None, None]

        min_year = self.reach.dates[0].year
        years = range(
            min_year, min_year + len(self.reach.IHA_nat["Group1"]["mean_january"])
        )
        groups = self.reach.scenarios[0]._require_iari().groups.keys()

        for g_idx, group in enumerate(groups):
            plt.figure()

            for i, scenario in enumerate(self.reach.scenarios):
                scenario_iari = scenario._require_iari()
                plt.plot(
                    years,
                    scenario_iari.groups[group],
                    color=self.scenario_colors[i],
                    label=scenario.name,
                )

            plt.title(f"{self.reach.name} - IARI Values Comparison - {group}")
            plt.xlabel("Year")
            plt.ylabel("IARI Value")
            plt.xlim(min(years), max(years))
            years_list = list(years)
            plt.xticks(years_list, [str(y) for y in years_list], rotation=45)
            plt.grid(True)
            plt.legend()

            # Set y-axis limits if provided
            if ylims[g_idx] is not None:
                plt.ylim(ylims[g_idx])
            else:
                plt.ylim(bottom=0)

            if save:
                plt.savefig(
                    os.path.join(self.output_dir, f"{group}_IARI_Comparison.png"),
                    bbox_inches="tight",
                )
        return plt.gca()

    def plot_iha_parameters(self, save: bool = False) -> Axes:
        """
        Plot IHA parameter comparisons for all parameters.

        Parameters
        ----------
        save : bool, default=False
            Whether to save the plots to files
        """
        min_year = self.reach.dates[0].year
        num_years = len(self.reach.IHA_nat["Group1"]["mean_january"])
        years_nat = range(min_year, min_year + num_years)
        years_alt = range(years_nat[-1] + 1, years_nat[-1] + 1 + num_years)
        years = list(years_nat) + list(years_alt)

        first_scenario_iha = self.reach.scenarios[0]._require_iha()
        for IHA_group_name, IHA_group in first_scenario_iha.items():
            for indicator in IHA_group:
                natural_values = self.reach.IHA_nat[IHA_group_name][indicator]

                plt.figure()
                plt.plot(
                    years_nat,
                    natural_values,
                    color="tab:blue",
                    label="Natural",
                )

                # Calculate and plot percentiles for natural flow.
                # The 25th and 75th percentiles define a band that represents the
                # natural inter-annual variability of this indicator, which is used in the computation of the IARI index and serves as reference range when visually comparing scenario results.
                p25 = float(np.percentile(natural_values, 25))
                p75 = float(np.percentile(natural_values, 75))

                # Plot percentile lines across the entire time range
                plt.axhline(
                    y=p25,
                    color="tab:blue",
                    linestyle="--",
                )
                plt.axhline(y=p75, color="tab:blue", linestyle="--")

                for j, scenario in enumerate(self.reach.scenarios):
                    scenario_iha = scenario._require_iha()
                    plt.plot(
                        years_alt,
                        scenario_iha[IHA_group_name][indicator],
                        label=scenario.name,
                        color=self.scenario_colors[j],
                    )

                plt.title(f"{self.reach.name} - IHA - {indicator}")
                plt.xlabel("Year")
                plt.ylabel("IHA Value")
                plt.xlim(min(years), max(years))
                plt.xticks(years, [str(y) for y in years], rotation=45)
                plt.grid(True)
                plt.legend()

                if save:
                    plt.savefig(
                        os.path.join(
                            self._ensure_iha_dir(), f"{indicator}_IHA_Comparison.png"
                        ),
                        bbox_inches="tight",
                    )
        return plt.gca()

    def plot_iari_summary(self, save: bool = False) -> Axes:
        """
        Create a summary bar plot of IARI indices for all scenarios.

        Parameters
        ----------
        save : bool, default=False
            Whether to save the plot to file
        """
        groups = ["Group1", "Group2", "Group3", "Group4", "Group5"]
        n_scenarios = len(self.reach.scenarios)

        # Calculate mean IARI values for each group and scenario
        means = {
            scenario.name: [
                np.mean(scenario._require_iari().groups[group]) for group in groups
            ]
            for scenario in self.reach.scenarios
        }
        for scenario in self.reach.scenarios:
            aggregated_iari = scenario._require_iari().aggregated
            means[scenario.name].append(np.mean(aggregated_iari))

        # Create grouped bar plot
        width = 0.8 / n_scenarios
        fig, ax = plt.subplots()

        bar_labels = groups + ["Aggregated"]
        for i, (scenario_name, values) in enumerate(means.items()):
            x = np.arange(len(bar_labels)) + (i - n_scenarios / 2 + 0.5) * width
            ax.bar(x, values, width, label=scenario_name, color=self.scenario_colors[i])

        ax.set_ylabel("Mean IARI Value")
        ax.set_title(f"{self.reach.name} IARI Summary")
        ax.set_xticks(np.arange(len(bar_labels)))
        ax.set_xticklabels(bar_labels)
        ax.legend()
        ax.grid(True, alpha=0.3)

        if save:
            plt.savefig(
                os.path.join(self.output_dir, "IARI_Summary.png"), bbox_inches="tight"
            )
        return plt.gca()

    def plot_nIHA_summary(self, save: bool = False) -> Axes:
        """
        Create a summary bar plot of normalized IHA indices for all scenarios.

        Parameters
        ----------
        save : bool, default=False
            Whether to save the plot to file
        """
        groups = ["Group1", "Group2", "Group3", "Group4", "Group5"]
        n_scenarios = len(self.reach.scenarios)

        # Calculate mean nIHA values for each group and scenario
        means = {
            scenario.name: [
                np.mean(scenario._require_normalized_iha().groups[group])
                for group in groups
            ]
            for scenario in self.reach.scenarios
        }
        for scenario in self.reach.scenarios:
            aggregated_niha = scenario._require_normalized_iha().aggregated
            means[scenario.name].append(np.mean(aggregated_niha))

        # Create grouped bar plot
        width = 0.8 / n_scenarios
        fig, ax = plt.subplots()

        bar_labels = groups + ["Aggregated"]
        for i, (scenario_name, values) in enumerate(means.items()):
            x = np.arange(len(bar_labels)) + (i - n_scenarios / 2 + 0.5) * width
            ax.bar(x, values, width, label=scenario_name, color=self.scenario_colors[i])

        ax.set_ylabel("Mean nIHA Value")
        ax.set_title(f"{self.reach.name} nIHA Summary")
        ax.set_xticks(np.arange(len(bar_labels)))
        ax.set_xticklabels(bar_labels)
        ax.legend()
        ax.grid(True, alpha=0.3)

        if save:
            plt.savefig(
                os.path.join(self.output_dir, "nIHA_Summary.png"), bbox_inches="tight"
            )
        return plt.gca()

    def plot_iha_boxplots(self, save: bool = False) -> Axes:
        """
        Create boxplot comparisons for IHA parameters across scenarios.

        Parameters
        ----------
        save : bool, default=False
            Whether to save the plots to files
        """
        first_scenario_iha = self.reach.scenarios[0]._require_iha()
        for IHA_group_name, IHA_group in first_scenario_iha.items():
            for indicator in IHA_group:
                plt.figure()

                data = [self.reach.IHA_nat[IHA_group_name][indicator]]
                labels = ["Natural"]

                for scenario in self.reach.scenarios:
                    scenario_iha = scenario._require_iha()
                    data.append(scenario_iha[IHA_group_name][indicator])
                    labels.append(scenario.name)

                plt.boxplot(data, tick_labels=labels)
                plt.title(f"{self.reach.name} - IHA Distribution - {indicator}")
                plt.ylabel("Value")
                plt.grid(True, alpha=0.3)
                plt.xticks(rotation=45)

                if save:
                    plt.savefig(
                        os.path.join(
                            self._ensure_iha_dir(), f"{indicator}_boxplot.png"
                        ),
                        bbox_inches="tight",
                    )
        return plt.gca()

    def plot_relative_deviations(self, save: bool = False) -> Axes:
        """
        Plot relative deviations of IHA parameters from natural flow.

        Parameters
        ----------
        save : bool, default=False
            Whether to save the plots to files
        """
        first_scenario_iha = self.reach.scenarios[0]._require_iha()
        for IHA_group_name, IHA_group in first_scenario_iha.items():
            for indicator in IHA_group:
                plt.figure()
                natural_values = self.reach.IHA_nat[IHA_group_name][indicator]

                min_year = self.reach.dates[0].year
                years = range(min_year, min_year + len(natural_values))

                for scenario in self.reach.scenarios:
                    scenario_iha = scenario._require_iha()
                    scenario_values = scenario_iha[IHA_group_name][indicator]
                    relative_dev = (
                        (scenario_values - natural_values) / natural_values * 100
                    )

                    plt.plot(
                        years,
                        relative_dev,
                        label=scenario.name,
                        color=self.scenario_colors[
                            self.reach.scenarios.index(scenario)
                        ],
                    )

                plt.title(
                    f"{self.reach.name} - Relative Deviation from Natural - {indicator}"
                )
                plt.xlabel("Year")
                plt.ylabel("Relative Deviation (%)")
                plt.xlim(min(years), max(years))
                years_list = list(years)
                plt.xticks(years_list, [str(y) for y in years_list], rotation=45)
                plt.grid(True)
                plt.legend()

                if save:
                    plt.savefig(
                        os.path.join(
                            self._ensure_iha_dir(),
                            f"{indicator}_relative_deviation.png",
                        ),
                        bbox_inches="tight",
                    )
        return plt.gca()

    def plot_cases_duration(self, save: bool = False) -> Axes:
        """
        Create a bar plot showing the duration percentage of each flow case for all scenarios.

        Flow cases are:
        - Case 1: Q ≤ Qreq (Natural flow is less than or equal to minimum release)
        - Case 2: Qreq < Q < Qreq + Qabs_max (Natural flow is between minimum release and maximum abstraction)
        - Case 3: Q ≥ Qreq + Qabs_max (Natural flow exceeds maximum abstraction capacity)

        Parameters
        ----------
        save : bool, default=False
            Whether to save the plot to file
        """
        # Ensure all scenarios have computed their cases_duration
        for scenario in self.reach.scenarios:
            if (
                not hasattr(scenario, "cases_duration")
                or scenario.cases_duration is None
            ):
                scenario.compute_Qrel()  # This will compute cases_duration as well

        n_scenarios = len(self.reach.scenarios)
        case_labels = [
            "Case 1\n(Qnat ≤ Qreq)",
            "Case 2\n(Qreq < Qnat < Qreq+Qabs_max)",
            "Case 3\n(Qnat ≥ Qreq+Qabs_max)",
        ]

        # Create a figure with appropriate size
        plt.figure()

        # Set up bar positions
        x = np.arange(len(case_labels))
        width = 0.8 / n_scenarios  # Width of bars

        # Plot bars for each scenario
        for i, scenario in enumerate(self.reach.scenarios):
            pos = x + (i - n_scenarios / 2 + 0.5) * width
            plt.bar(
                pos,
                [d * 100 for d in scenario.cases_duration],
                width,
                label=scenario.name,
                color=self.scenario_colors[i],
            )

        plt.ylabel("Duration (%)")
        plt.title(f"{self.reach.name} - Flow Cases Duration by Scenario")
        plt.xticks(x, case_labels)
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Add value labels on top of each bar
        for i, scenario in enumerate(self.reach.scenarios):
            pos = x + (i - n_scenarios / 2 + 0.5) * width
            for j, value in enumerate(scenario.cases_duration):
                plt.text(
                    pos[j],
                    value * 100,
                    f"{value*100:.0f}%",
                    horizontalalignment="center",
                    verticalalignment="bottom",
                )

        if save:
            plt.savefig(
                os.path.join(self.output_dir, "cases_duration.png"), bbox_inches="tight"
            )
        return plt.gca()

    def plot_cases_duration_month(self, month, save: bool = False) -> Axes:
        """
        Create a bar plot showing the duration percentage of each flow case for all scenarios for a specific month.

        Parameters
        ----------
        month : int or str
            Month to plot (1-12 or month name, e.g., 'Jan', 'January')
        save : bool, default=False
            Whether to save the plot to file
        """
        # Convert month input to integer (1-12)
        if isinstance(month, str):
            month_strs = [
                "jan",
                "feb",
                "mar",
                "apr",
                "may",
                "jun",
                "jul",
                "aug",
                "sep",
                "oct",
                "nov",
                "dec",
            ]
            month_lower = month.strip().lower()[:3]
            if month_lower in month_strs:
                month_num = month_strs.index(month_lower) + 1
            else:
                raise ValueError(f"Invalid month string: {month}")
        elif isinstance(month, int) and 1 <= month <= 12:
            month_num = month
        else:
            raise ValueError(
                "month must be an integer (1-12) or a valid month name string"
            )

        n_scenarios = len(self.reach.scenarios)
        case_labels = [
            "Case 1\n(Q ≤ Qreq)",
            "Case 2\n(Qreq < Q < Qreq+Qabs_max)",
            "Case 3\n(Q ≥ Qreq+Qabs_max)",
        ]

        plt.figure()
        x = np.arange(len(case_labels))
        width = 0.8 / n_scenarios

        for i, scenario in enumerate(self.reach.scenarios):
            # Use the new method from Scenario for robust calculation
            durations = scenario.cases_duration_for_month(month_num)
            pos = x + (i - n_scenarios / 2 + 0.5) * width
            plt.bar(
                pos,
                [d * 100 for d in durations],
                width,
                label=scenario.name,
                color=self.scenario_colors[i],
            )
            # Add value labels
            for j, value in enumerate(durations):
                plt.text(
                    pos[j],
                    value * 100,
                    f"{value*100:.0f}%",
                    horizontalalignment="center",
                    verticalalignment="bottom",
                )

        plt.ylabel("Duration (%)")
        plt.title(
            f"{self.reach.name} - Flow Cases Duration by Scenario - {month if isinstance(month, str) else month_num}"
        )
        plt.xticks(x, case_labels)
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save:
            plt.savefig(
                os.path.join(
                    self.output_dir,
                    f"cases_duration_month_{month if isinstance(month, str) else month_num}.png",
                ),
                bbox_inches="tight",
            )
        return plt.gca()

    def plot_monthly_abstraction(self, save: bool = False) -> Axes:
        """
        Create a bar plot showing the average monthly abstracted volumes for each scenario.

        Parameters
        ----------
        save : bool, default=False
            Whether to save the plot to file
        """
        # Ensure all scenarios have computed their abstracted volumes
        for scenario in self.reach.scenarios:
            if (
                not hasattr(scenario, "monthly_abs_volumes")
                or scenario.monthly_abs_volumes is None
            ):
                scenario.compute_natural_abstracted_volumes()

        # Set up plot
        plt.figure()
        months = [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]
        x = np.arange(len(months))
        width = 0.8 / (len(self.reach.scenarios) + 1)  # +1 for natural flow

        # Plot natural flow volumes as reference
        plt.bar(
            x,
            self.reach.scenarios[0].monthly_nat_volumes / 1e6,
            width,
            label="Natural",
            color="tab:blue",
            alpha=0.3,
        )

        # Plot abstracted volumes for each scenario
        for i, scenario in enumerate(self.reach.scenarios):
            pos = x + (i + 1) * width - 0.4
            plt.bar(
                pos,
                scenario.monthly_abs_volumes / 1e6,
                width,
                label=f"{scenario.name}",
                color=self.scenario_colors[i],
            )

        # Customize plot
        plt.xlabel("Month")
        plt.ylabel("Volume (million m³)")
        plt.title(f"{self.reach.name} - Monthly-averaged Abstracted Water Volumes")
        plt.xticks(x, months)
        plt.legend(title="Scenario")
        plt.grid(True, alpha=0.3)

        if save:
            plt.savefig(
                os.path.join(self.output_dir, "monthly_abstraction.png"),
                bbox_inches="tight",
            )
        return plt.gca()

    def plot_iari_vs_volume(self, save: bool = False) -> Axes:
        """
        Create a scatter plot showing the relationship between abstracted volumes and IARI indices.
        Each scenario is shown with error bars representing standard deviations.

        X-axis shows ecohydrological quality (1 - IARI)
        Y-axis shows normalized abstracted volume (Vturb/Vnat)

        Parameters
        ----------
        save : bool, default=False
            Whether to save the plot to file
        """
        plt.figure()

        # For each scenario except natural flow
        for i, scenario in enumerate(self.reach.scenarios):
            # Calculate IARI statistics
            iari_values = scenario._require_iari().aggregated
            iari_median = np.median(iari_values)
            iari_std = np.std(iari_values)

            # Calculate volume statistics
            vol_values = scenario.yearly_abs_volumes / scenario.yearly_nat_volumes
            vol_median = np.median(vol_values)
            vol_std = np.std(vol_values)

            # Plot error bars and point
            plt.errorbar(
                1 - iari_median,
                vol_median,
                xerr=iari_std,
                yerr=vol_std,
                fmt="^",  # marker style
                color=self.scenario_colors[i],
                label=scenario.name,
                capsize=5,
                elinewidth=1,
                markersize=10,
            )

        plt.xlabel(r"Ecohydrological quality $(1 - IARI)$ [-]")
        plt.ylabel(r"Normalized abstracted volume $V_{der}/V_{nat}$ [-]")
        plt.grid(True)
        plt.legend()
        plt.title(f"{self.reach.name} - IARI vs Abstracted Volume")

        if save:
            plt.savefig(
                os.path.join(self.output_dir, "iari_vs_volume.png"),
                bbox_inches="tight",
            )
        return plt.gca()

    def plot_hq_curves(
        self,
        save: bool = False,
        xlim: Optional[float] = None,
        rule_min: Optional[float] = None,
        rule_max: Optional[float] = None,
        rule_name: str = "DMV",
    ) -> Axes:
        """
        Plot all HQ curves of the reach.

        Parameters
        ----------
        save : bool, default=False
            Whether to save the plot to file
        xlim : float or None, default=None
            Optional maximum value for x-axis. If provided, x-axis limits are [0, xlim].
        rule_min : float or None, default=None
            Lower bound of optional rule range marker.
        rule_max : float or None, default=None
            Upper bound of optional rule range marker.
        rule_name : str, default="DMV"
            Label used for the optional rule range markers.
        """

        plt.figure(figsize=(10, 6))

        for species in self.reach.get_list_available_HQ_curves():
            curve = self.reach.get_HQ_curve(curve_name=species)
            plt.plot(curve["DIS"], curve[species], label=f"{species}")

        # LABELS AND TITLE
        plt.xlabel(r"Flow discharge $[\mathrm{m}^3/\mathrm{s}]$")
        plt.ylabel(r"Available habitat area $[\mathrm{m}^2]$")
        plt.title("Habitat-Discharge (HQ) curves")

        if xlim is not None:
            plt.xlim(0, xlim)

        if rule_min is not None and rule_max is not None:
            plt.axvline(
                x=rule_min,
                color="tab:gray",
                linestyle="--",
                label=f"{rule_name} range: {rule_min}-{rule_max} m³/s",
            )
            plt.axvline(x=rule_max, color="tab:gray", linestyle="--")

        # Show the plot
        plt.grid(True)
        plt.legend()

        if save:
            plt.savefig(
                os.path.join(self.output_dir, "hq_curves.png"),
                bbox_inches="tight",
            )
        return plt.gca()

    def plot_habitat_timeseries(
        self,
        species: str,
        save: bool = False,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
    ) -> Axes:
        """
        Plot habitat time series for a species across all scenarios.

        Parameters
        ----------
        species : str
            Species to plot
        save : bool, default=False
            Whether to save the plot to file
        start_year : int or None, default=None
            Start year for the plot range. If None, the first year in reach dates is used.
        end_year : int or None, default=None
            End year for the plot range. If None, the last year in reach dates is used.
        """

        plt.figure()
        dates = np.array(self.reach.dates)
        for i, scenario in enumerate(self.reach.scenarios):
            plt.plot(
                dates,
                scenario.IH[species].H_alt,
                label=f"{scenario.name} - {species}",
                color=self.scenario_colors[i],
            )

        plt.plot(
            dates, scenario.IH[species].H_ref, label=f"Reference Q", color="tab:blue"
        )

        plt.xlim(
            datetime(start_year if start_year else self.reach.dates[0].year, 1, 1),
            datetime(end_year if end_year else self.reach.dates[-1].year, 12, 31),
        )

        plt.xticks(rotation=45)
        plt.xlabel("Date")
        plt.ylabel("Available Habitat Area [%]")
        plt.title(f"{self.reach.name} - Habitat Time Series - {species}")
        plt.grid(True)
        plt.legend()
        if save:
            plt.savefig(
                os.path.join(self.output_dir, f"habitat_timeseries_{species}.png"),
                bbox_inches="tight",
            )
        return plt.gca()

    def plot_ucut_curves(
        self,
        species: str,
        save: bool = False,
    ) -> Axes:
        """
        Plot UCUT curves for a species across all scenarios.

        Parameters
        ----------
        species : str
            Species to plot
        save : bool, default=False
            Whether to save the plot to file
        """

        plt.figure()
        for i, scenario in enumerate(self.reach.scenarios):
            plt.plot(
                scenario.IH[species].ucut_alt.cum_freq * 100,
                scenario.IH[species].ucut_alt.durations,
                label=f"{scenario.name} - {species}",
                color=self.scenario_colors[i],
            )

        plt.plot(
            scenario.IH[species].ucut_ref.cum_freq * 100,
            scenario.IH[species].ucut_ref.durations,
            label=f"Reference Q",
            color="tab:blue",
        )

        plt.xticks(rotation=45)
        plt.xlabel("Cumulative continuous duration [%]")
        plt.ylabel("Continuous days below threshold [days]")
        plt.title(f"{self.reach.name} - UCUT - {species}")
        plt.grid(True)
        plt.legend()
        if save:
            plt.savefig(
                os.path.join(
                    self.output_dir,
                    f"habitat_timeseries_{species}_{self.reach.name}.png",
                ),
                bbox_inches="tight",
            )
        return plt.gca()

    def plot_ih_vs_volume(self, save: bool = False) -> Axes:
        """
        Create a scatter plot showing the relationship between abstracted volumes and IH index for a selected species.
        Each scenario is shown with error bars representing standard deviations of volumes.

        X-axis shows Habitat Index (IH)
        Y-axis shows normalized abstracted volume (Vturb/Vnat)

        Parameters
        ----------
        save : bool, default=False
            Whether to save the plot to file
        """
        plt.figure()

        # Create color mapping for species
        all_species = set()
        for scenario in self.reach.scenarios:
            all_species.update(scenario.IH.keys())

        species_colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(all_species)))
        species_color_map = {
            species: species_colors[i] for i, species in enumerate(sorted(all_species))
        }
        species_plotted = set()  # Track which species have been added to legend

        for i, scenario in enumerate(self.reach.scenarios):
            # Calculate IH statistics
            ih_values = []
            for species in scenario.IH.keys():
                ih_values.append(scenario.IH[species].IH)

            # Calculate volume statistics
            vol_values = scenario.yearly_abs_volumes / scenario.yearly_nat_volumes
            vol_median = np.median(vol_values)
            vol_std = np.std(vol_values)

            # Plot error bars and point
            plt.errorbar(
                min(ih_values),
                vol_median,
                yerr=vol_std,
                xerr=np.array([[0], [max(ih_values) - min(ih_values)]]),
                fmt="^",  # marker style
                color=self.scenario_colors[i],
                label=scenario.name,
                capsize=5,
                elinewidth=1,
                markersize=10,
            )

            # plot a point for each species IH
            for species in scenario.IH.keys():
                plt.plot(
                    scenario.IH[species].IH,
                    vol_median,
                    "o",
                    color=species_color_map[species],
                    alpha=0.7,
                    markersize=8,
                    label=species if species not in species_plotted else None,
                )
                species_plotted.add(species)

        plt.xlabel(r"Habitat Index $(IH)$ [-]")
        plt.ylabel(r"Normalized abstracted volume $V_{der}/V_{nat}$ [-]")
        plt.grid(True)
        plt.legend()
        plt.title(f"{self.reach.name} - IH vs Abstracted Volume")

        if save:
            plt.savefig(
                os.path.join(self.output_dir, "ih_vs_volume.png"),
                bbox_inches="tight",
            )
        return plt.gca()

    def plot_sediment_budget_vs_volume(self, save: bool = False) -> Axes:
        """Plot normalized annual sediment budgets against abstracted volumes.

        Each scenario's annual sediment budget is divided by the natural budget
        for the same year. The plot shows the mean and standard deviation of
        these annual ratios, alongside the scenario's normalized abstracted
        volume statistics.

        Parameters
        ----------
        save : bool, default=False
            Whether to save the plot to file.
        """
        if not self.reach.scenarios:
            raise ValueError("The reach has no scenarios to plot.")

        scenario_budgets = [
            (scenario, scenario._require_annual_sediment_budget())
            for scenario in self.reach.scenarios
        ]
        if self.reach.natural_annual_sediment_budget is None:
            self.reach.compute_natural_sediment_budget()
        natural_budget = self.reach._require_natural_annual_sediment_budget()

        plt.figure()
        for i, (scenario, scenario_budget) in enumerate(scenario_budgets):
            years = scenario_budget.index.intersection(natural_budget.index)
            if len(years) == 0:
                raise ValueError(
                    f"Scenario '{scenario.name}' has no annual sediment budget years in common with the natural budget."
                )

            natural_values = natural_budget.loc[years, "Qs_total"].to_numpy(dtype=float)
            scenario_values = scenario_budget.loc[years, "Qs_total"].to_numpy(
                dtype=float
            )
            valid = (
                np.isfinite(natural_values)
                & np.isfinite(scenario_values)
                & (natural_values > 0)
            )
            if not np.any(valid):
                raise ValueError(
                    f"Scenario '{scenario.name}' has no years with a positive natural sediment budget."
                )

            normalized_sediment_budget = scenario_values[valid] / natural_values[valid]
            sediment_mean = np.mean(normalized_sediment_budget)
            sediment_std = np.std(normalized_sediment_budget)

            if (
                getattr(scenario, "yearly_abs_volumes", None) is None
                or getattr(scenario, "yearly_nat_volumes", None) is None
            ):
                scenario.compute_natural_abstracted_volumes()
            volume_values = scenario.yearly_abs_volumes / scenario.yearly_nat_volumes

            plt.errorbar(
                sediment_mean,
                np.median(volume_values),
                xerr=sediment_std,
                yerr=np.std(volume_values),
                fmt="^",
                color=self.scenario_colors[i],
                label=scenario.name,
                capsize=5,
                elinewidth=1,
                markersize=10,
            )

        plt.xlabel("Normalized sediment budget (scenario / natural) [-]")
        plt.ylabel(r"Normalized abstracted volume $V_{der}/V_{nat}$ [-]")
        plt.grid(True)
        plt.legend()
        plt.title(f"{self.reach.name} - Sediment Budget vs Abstracted Volume")

        if save:
            plt.savefig(
                os.path.join(self.output_dir, "sediment_budget_vs_volume.png"),
                bbox_inches="tight",
            )
        return plt.gca()

    def plot_nIHA_vs_volume(self, save: bool = False) -> Axes:
        """
        Create a scatter plot showing the relationship between abstracted volumes and nIHA indexes.
        Each scenario is shown with error bars representing standard deviations.

        X-axis shows ecohydrological quality (-nIHA)

        Y-axis shows normalized abstracted volume (Vturb/Vnat)

        Parameters
        ----------
        save : bool, default=False
            Whether to save the plot to file
        """
        plt.figure()

        # For each scenario except natural flow
        for i, scenario in enumerate(self.reach.scenarios):
            # Calculate nIHA statistics
            nIHA_values = scenario._require_normalized_iha().aggregated
            nIHA_median = np.median(nIHA_values)
            nIHA_std = np.std(nIHA_values)

            # Calculate volume statistics
            vol_values = scenario.yearly_abs_volumes / scenario.yearly_nat_volumes
            vol_median = np.median(vol_values)
            vol_std = np.std(vol_values)

            # Plot error bars and point
            plt.errorbar(
                -nIHA_median,
                vol_median,
                xerr=nIHA_std,
                yerr=vol_std,
                fmt="^",  # marker style
                color=self.scenario_colors[i],
                label=scenario.name,
                capsize=5,
                elinewidth=1,
                markersize=10,
            )

        plt.xlabel(r"Ecohydrological quality $(-nIHA)$ [-]")
        plt.ylabel(r"Normalized abstracted volume $V_{der}/V_{nat}$ [-]")
        plt.grid(True)
        plt.legend()
        plt.title(f"{self.reach.name} - nIHA vs Abstracted Volume")

        if save:
            plt.savefig(
                os.path.join(self.output_dir, "niha_vs_volume.png"),
                bbox_inches="tight",
            )
        return plt.gca()

    def plot_sediment_load_total(
        self,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        log_scale: bool = True,
        save: bool = False,
    ) -> Axes:
        """
        Plot total sediment load (Qs_total) over time for all scenarios.

        Parameters
        ----------
        start_date : str or datetime, optional
            Start date for the plot
        end_date : str or datetime, optional
            End date for the plot
        log_scale : bool, default=True
            Use log scale for y-axis
        save : bool, default=False
            Whether to save the plot
        """
        mask = _compute_date_mask(self.reach.dates, start_date, end_date)

        sediment_loads = [
            (scenario, scenario._require_sediment_load_df())
            for scenario in self.reach.scenarios
        ]

        plt.figure()
        for i, (scenario, sediment_load) in enumerate(sediment_loads):
            Qs_total = sediment_load["Qs_total"].to_numpy(dtype=float)
            plt.plot(
                np.array(self.reach.dates)[mask],
                Qs_total[mask],
                label=scenario.name,
                color=self.scenario_colors[i],
            )

        if log_scale:
            plt.yscale("log")
        plt.xlabel("Date")
        plt.ylabel("Total Sediment Load (m³/s)")
        plt.title(f"{self.reach.name} - Total Sediment Load")
        plt.grid(True)
        plt.legend()

        if save:
            plt.savefig(
                os.path.join(self.output_dir, "sediment_load_total.png"),
                bbox_inches="tight",
            )
        return plt.gca()

    def plot_annual_sediment_budget_by_class(
        self,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
        save: bool = False,
    ) -> Axes:
        """
        Plot the annual sediment budget per grain-size class as year-by-size heatmaps.

        Each scenario gets a panel with one column per year and one row per phi
        class, using the table stored in ``scenario.annual_sediment_budget``
        (see ``Scenario.compute_annual_sediment_budget``). The colour scale is
        shared across scenarios and expressed in the units of the stored budget.

        Parameters
        ----------
        start_year : int, optional
            First year to plot
        end_year : int, optional
            Last year to plot
        save : bool, default=False
            Whether to save the plot

        Returns
        -------
        Axes
            The first scenario panel. The other panels and colorbar are
            available from ``ax.figure.axes``.
        """
        budgets = [
            (scenario, scenario._require_annual_sediment_budget())
            for scenario in self.reach.scenarios
        ]
        if not budgets:
            raise ValueError("The reach has no scenarios to plot.")

        phi_cols = [
            column for column in budgets[0][1].columns if column.startswith("Qs_phi_")
        ]
        if not phi_cols:
            raise ValueError("Annual budgets contain no per-class data.")

        diameters_mm = np.array(
            [2 ** (-float(column[len("Qs_phi_") :])) for column in phi_cols]
        )
        size_order = np.argsort(diameters_mm)
        phi_cols = [phi_cols[index] for index in size_order]
        diameters_mm = diameters_mm[size_order]

        # Keep only the classes overlapping the user-provided grain size range
        grain_size_data = getattr(self.reach, "grain_size_data", None)
        if grain_size_data is not None:
            d_min = float(grain_size_data["di[mm]"].min())
            d_max = float(grain_size_data["di[mm]"].max())
            keep = (diameters_mm * np.sqrt(2) >= d_min) & (
                diameters_mm / np.sqrt(2) <= d_max
            )
            if keep.any():
                phi_cols = [c for c, k in zip(phi_cols, keep) if k]
                diameters_mm = diameters_mm[keep]
        # Every class gets the same vertical space (unit-height rows)
        row_edges = np.arange(len(diameters_mm) + 1)

        filtered_budgets = []
        for scenario, budget in budgets:
            df = budget
            if start_year is not None:
                df = df.loc[df.index >= start_year]
            if end_year is not None:
                df = df.loc[df.index <= end_year]
            if df.empty:
                raise ValueError("No annual budget falls within the requested years.")
            filtered_budgets.append((scenario, df))

        max_budget = max(
            float(df[phi_cols].to_numpy(dtype=float).max())
            for _, df in filtered_budgets
        )
        if max_budget == 0:
            max_budget = 1.0

        fig, axes = plt.subplots(
            len(filtered_budgets),
            1,
            figsize=(12, 3.5 * len(filtered_budgets)),
            sharex=True,
            sharey=True,
            squeeze=False,
        )
        axes = axes[:, 0]
        for ax, (scenario, df) in zip(axes, filtered_budgets):
            years = np.asarray(df.index, dtype=float)
            year_edges = np.append(years - 0.5, years[-1] + 0.5)
            mesh = ax.pcolormesh(
                year_edges,
                row_edges,
                df[phi_cols].to_numpy(dtype=float).T,
                shading="flat",
                cmap="Oranges",
                vmin=0,
                vmax=max_budget,
            )
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_yticks(row_edges[:-1] + 0.5)
            ax.set_yticklabels([f"{d:.1e}" for d in diameters_mm])
            ax.set_title(scenario.name)
            ax.set_ylabel("Grain diameter (mm)")

        axes[-1].set_xlabel("Year")
        fig.suptitle(f"{self.reach.name} - Annual Sediment Budget by Grain-Size Class")
        # Reserve the right-hand strip of the figure for the colorbar
        fig.tight_layout(rect=(0, 0, 0.9, 0.96))
        cbar_ax = fig.add_axes((0.92, 0.12, 0.02, 0.76))
        fig.colorbar(mesh, cax=cbar_ax, label="Annual sediment budget (m³/year)")

        if save:
            fig.savefig(
                os.path.join(self.output_dir, "annual_sediment_budget_by_class.png"),
                bbox_inches="tight",
            )
        return axes[0]
