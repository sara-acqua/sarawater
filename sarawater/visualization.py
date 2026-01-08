"""
This module provides plotting functionality for comparing scenarios in a reach.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Optional, Union
from datetime import datetime

from sarawater.reach import Reach
from sarawater.scenarios import Scenario


class ReachPlotter:
    """A class for plotting and comparing scenarios in a reach."""

    def __init__(
        self,
        reach: Reach,
        output_dir: Optional[str] = "outputs",
        scenario_colors: List[str] = [
            "tab:red",
            "tab:orange",
            "tab:green",
            "tab:purple",
            "tab:brown",
            "tab:pink",
            "tab:gray",
            "tab:olive",
            "tab:cyan",
        ],
    ):
        """
        Initialize a ReachPlotter instance.

        Parameters
        ----------
        reach : Reach
            The reach object containing scenarios to plot
        output_dir : str or None, optional
            Directory where to save the plots. By default "outputs". Set to None to prevent the directory from being created. Note that plotting methods need to be called with save=True to save the plots.
        scenario_colors : list of str, optional
            List of colors to use for each scenario in the plots. Default is a set of distinct tab colors.
        """
        self.reach = reach
        self.scenario_colors = scenario_colors
        self.output_dir = output_dir
        if output_dir is not None:
            os.makedirs(self.output_dir, exist_ok=True)

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
    ) -> None:
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
        save : bool, default=True
            Whether to save the plot to file
        plot_Qnat : bool, default=True
            Whether to plot the natural flow (Qnat)
        """
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)

        if start_date and end_date:
            mask = [(dt >= start_date) and (dt <= end_date) for dt in self.reach.dates]
        else:
            start_date = self.reach.dates[0]
            end_date = self.reach.dates[-1]
            mask = [True] * len(self.reach.dates)

        plt.figure()
        for color, scenario in zip(self.scenario_colors, self.reach.scenarios):
            plt.plot(
                np.array(self.reach.dates)[mask],
                scenario.QS[mask],
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
        self, save: bool = False, ylims: list = [None, None, None, None, None]
    ) -> None:
        """
        Plot IARI values comparison for each group.

        Parameters
        ----------
        ylims : list, default=None for all groups
            Y-axis limit for each group. If None, the limit is set automatically.
        save : bool, default=True
            Whether to save the plots to files
        """
        min_year = self.reach.dates[0].year
        years = range(
            min_year, min_year + len(self.reach.IHA_nat["Group1"]["mean_january"])
        )
        groups = self.reach.scenarios[0].IARI["groups"].keys()

        for g_idx, group in enumerate(groups):
            plt.figure()

            for i, scenario in enumerate(self.reach.scenarios):
                plt.plot(
                    years,
                    scenario.IARI["groups"][group],
                    color=self.scenario_colors[i],
                    label=scenario.name,
                )

            plt.title(f"{self.reach.name} - IARI Values Comparison - {group}")
            plt.xlabel("Year")
            plt.ylabel("IARI Value")
            plt.xlim(min(years), max(years))
            plt.xticks(years, years, rotation=45)
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

    def plot_iha_parameters(self, save: bool = False) -> None:
        """
        Plot IHA parameter comparisons for all parameters.

        Parameters
        ----------
        save : bool, default=True
            Whether to save the plots to files
        """
        min_year = self.reach.dates[0].year
        years = range(
            min_year, min_year + len(self.reach.IHA_nat["Group1"]["mean_january"])
        )

        for IHA_group_name, IHA_group in self.reach.scenarios[0].IHA.items():
            for indicator in IHA_group:
                plt.figure()
                plt.plot(
                    years,
                    self.reach.IHA_nat[IHA_group_name][indicator],
                    label="Natural",
                )

                for j, scenario in enumerate(self.reach.scenarios):
                    plt.plot(
                        years,
                        scenario.IHA[IHA_group_name][indicator],
                        label=scenario.name,
                        color=self.scenario_colors[j],
                    )

                plt.title(f"{self.reach.name} - IHA - {indicator}")
                plt.xlabel("Year")
                plt.ylabel("IHA Value")
                plt.xlim(min(years), max(years))
                plt.xticks(years, years, rotation=45)
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

    def plot_iari_summary(self, save: bool = False) -> None:
        """
        Create a summary bar plot of IARI indices for all scenarios.

        Parameters
        ----------
        save : bool, default=True
            Whether to save the plot to file
        """
        groups = ["Group1", "Group2", "Group3", "Group4", "Group5"]
        n_scenarios = len(self.reach.scenarios)

        # Calculate mean IARI values for each group and scenario
        means = {
            scenario.name: [np.mean(scenario.IARI["groups"][group]) for group in groups]
            for scenario in self.reach.scenarios
        }
        for scenario in self.reach.scenarios:
            means[scenario.name].append(np.mean(scenario.IARI["aggregated"]))

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

    def plot_nIHA_summary(self, save: bool = False) -> None:
        """
        Create a summary bar plot of normalized IHA indices for all scenarios.

        Parameters
        ----------
        save : bool, default=True
            Whether to save the plot to file
        """
        groups = ["Group1", "Group2", "Group3", "Group4", "Group5"]
        n_scenarios = len(self.reach.scenarios)

        # Calculate mean nIHA values for each group and scenario
        means = {
            scenario.name: [
                np.mean(scenario.normalized_IHA["groups"][group]) for group in groups
            ]
            for scenario in self.reach.scenarios
        }
        for scenario in self.reach.scenarios:
            means[scenario.name].append(np.mean(scenario.normalized_IHA["aggregated"]))

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

    def plot_iha_boxplots(self, save: bool = False) -> None:
        """
        Create boxplot comparisons for IHA parameters across scenarios.

        Parameters
        ----------
        save : bool, default=True
            Whether to save the plots to files
        """
        for IHA_group_name, IHA_group in self.reach.scenarios[0].IHA.items():
            for indicator in IHA_group:
                plt.figure()

                data = [self.reach.IHA_nat[IHA_group_name][indicator]]
                labels = ["Natural"]

                for scenario in self.reach.scenarios:
                    data.append(scenario.IHA[IHA_group_name][indicator])
                    labels.append(scenario.name)

                plt.boxplot(data, labels=labels)
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

    def plot_relative_deviations(self, save: bool = False) -> None:
        """
        Plot relative deviations of IHA parameters from natural flow.

        Parameters
        ----------
        save : bool, default=True
            Whether to save the plots to files
        """
        for IHA_group_name, IHA_group in self.reach.scenarios[0].IHA.items():
            for indicator in IHA_group:
                plt.figure()
                natural_values = self.reach.IHA_nat[IHA_group_name][indicator]

                min_year = self.reach.dates[0].year
                years = range(min_year, min_year + len(natural_values))

                for scenario in self.reach.scenarios:
                    scenario_values = scenario.IHA[IHA_group_name][indicator]
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
                plt.xticks(years, years, rotation=45)
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

    def plot_cases_duration(self, save: bool = False) -> None:
        """
        Create a bar plot showing the duration percentage of each flow case for all scenarios.

        Flow cases are:
        - Case 1: Q ≤ QR (Natural flow is less than or equal to minimum release)
        - Case 2: QR < Q < QR + Qab_max (Natural flow is between minimum release and maximum abstraction)
        - Case 3: Q ≥ QR + Qab_max (Natural flow exceeds maximum abstraction capacity)

        Parameters
        ----------
        save : bool, default=True
            Whether to save the plot to file
        """
        # Ensure all scenarios have computed their cases_duration
        for scenario in self.reach.scenarios:
            if (
                not hasattr(scenario, "cases_duration")
                or scenario.cases_duration is None
            ):
                scenario.compute_QS()  # This will compute cases_duration as well

        n_scenarios = len(self.reach.scenarios)
        case_labels = [
            "Case 1\n(Q ≤ QR)",
            "Case 2\n(QR < Q < QR+Qab_max)",
            "Case 3\n(Q ≥ QR+Qab_max)",
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

    def plot_cases_duration_month(self, month, save: bool = False) -> None:
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
            "Case 1\n(Q ≤ QR)",
            "Case 2\n(QR < Q < QR+Qab_max)",
            "Case 3\n(Q ≥ QR+Qab_max)",
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

    def plot_monthly_abstraction(self, save: bool = False) -> None:
        """
        Create a bar plot showing the average monthly abstracted volumes for each scenario.

        Parameters
        ----------
        save : bool, default=True
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

    def plot_iari_vs_volume(self, save: bool = False) -> None:
        """
        Create a scatter plot showing the relationship between abstracted volumes and IARI indices.
        Each scenario is shown with error bars representing standard deviations.

        X-axis shows ecohydrological quality (1 - IARI)
        Y-axis shows normalized abstracted volume (Vturb/Vnat)

        Parameters
        ----------
        save : bool, default=True
            Whether to save the plot to file
        """
        plt.figure()

        # For each scenario except natural flow
        for i, scenario in enumerate(self.reach.scenarios):
            # Calculate IARI statistics
            iari_values = scenario.IARI["aggregated"]
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
        xlim: float = None,
        rule_min: float = None,
        rule_max: float = None,
        rule_name: str = "DMV",
    ) -> None:
        """
        Plot all HQ curves of the reach.

        Parameters


        ----------
        save : bool, default=True
            Whether to save the plot to file
        """

        plt.figure(figsize=(10, 6))

        for species in self.reach.get_list_available_HQ_curves():
            curve = self.reach.get_HQ_curve(curve_name=species)
            plt.plot(curve["DIS"], curve[species], label=f"{species}")

        # LABELS AND TITLE
        plt.xlabel(r"Q $[\mathrm{m}^3/\mathrm{s}]$")
        plt.ylabel(r"Available area $[\mathrm{m}^2]$")
        plt.title("Habitat-Discharge (HQ) curves")

        if xlim:
            plt.xlim(0, xlim)

        if rule_min and rule_max:
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
        start_year: int = None,
        end_year: int = None,
    ) -> None:
        """
        Plot habitat time series for specific species and scenario

        Parameters
        ----------
        save : bool, default=False
            Whether to save the plot to file
        species : str
            Species to plot
        scenario : Scenario
            Scenario to plot
        start_year : int
            Start year for the plot range
        end_year : int
            End year for the plot range
        """

        plt.figure()
        for scenario in self.reach.scenarios:
            plt.plot(
                self.reach.dates,
                scenario.IH[species]["H_alt"],
                label=f"{scenario.name} - {species}",
                # color="tab:orange",
            )

        plt.plot(
            self.reach.dates,
            scenario.IH[species]["H_ref"],
            label=f"Reference Q",
            # color="tab:blue",
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

    def plot_ucut_curves(
        self,
        species: str,
        save: bool = False,
    ) -> None:
        """
        Plot ucut curves for specific species and all scenarios

        Parameters
        ----------
        save : bool, default=True
            Whether to save the plot to file
        species : str
            Species to plot
        scenario : Scenario
            Scenario to plot
        """

        plt.figure()
        for scenario in self.reach.scenarios:
            plt.plot(
                scenario.IH[species]["UCUT_cum_alt"],
                scenario.IH[species]["UCUT_events_alt"],
                label=f"{scenario.name} - {species}",
                # color="tab:orange",
            )

        plt.plot(
            scenario.IH[species]["UCUT_cum_ref"],
            scenario.IH[species]["UCUT_events_ref"],
            label=f"Reference Q",
            # color="tab:blue",
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
                    self.output_dir, f"habitat_timeseries_{species}_{scenario.name}.png"
                ),
                bbox_inches="tight",
            )

    def plot_ih_vs_volume(self, save: bool = False) -> None:
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

        species_colors = plt.cm.tab10(np.linspace(0, 1, len(all_species)))
        species_color_map = {
            species: species_colors[i] for i, species in enumerate(sorted(all_species))
        }
        species_plotted = set()  # Track which species have been added to legend

        for i, scenario in enumerate(self.reach.scenarios):
            # Calculate IH statistics
            ih_values = []
            for species in scenario.IH.keys():
                ih_values.append(scenario.IH[species]["IH"])

            # Calculate volume statistics
            vol_values = scenario.yearly_abs_volumes / scenario.yearly_nat_volumes
            vol_median = np.median(vol_values)
            vol_std = np.std(vol_values)

            # Plot error bars and point
            plt.errorbar(
                min(ih_values),
                vol_median,
                yerr=vol_std,
                xerr=np.array([[0], [max(ih_values)] - min(ih_values)]),
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
                    scenario.IH[species]["IH"],
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

    def plot_nIHA_vs_volume(self, save: bool = False) -> None:
        """
        Create a scatter plot showing the relationship between abstracted volumes and nIHA indexes.
        Each scenario is shown with error bars representing standard deviations.

        X-axis shows ecohydrological quality (1 - nIHA)
        Y-axis shows normalized abstracted volume (Vturb/Vnat)

        Parameters
        ----------
        save : bool, default=True
            Whether to save the plot to file
        """
        plt.figure()

        # For each scenario except natural flow
        for i, scenario in enumerate(self.reach.scenarios):
            # Calculate nIHA statistics
            nIHA_values = scenario.normalized_IHA["aggregated"]
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
    ) -> None:
        """
        Plot total sediment load (qS_total) over time for all scenarios.

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
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)

        if start_date and end_date:
            mask = [(dt >= start_date) and (dt <= end_date) for dt in self.reach.dates]
        else:
            mask = [True] * len(self.reach.dates)

        plt.figure()
        for i, scenario in enumerate(self.reach.scenarios):
            if not hasattr(scenario, "sediment_load"):
                continue
            plt.plot(
                np.array(self.reach.dates)[mask],
                scenario.sediment_load["qS_total"].values[mask],
                label=scenario.name,
                color=self.scenario_colors[i],
            )

        if log_scale:
            plt.yscale("log")
        plt.xlabel("Date")
        plt.ylabel("Total Sediment Load (kg/s)")
        plt.title(f"{self.reach.name} - Total Sediment Load")
        plt.grid(True)
        plt.legend()

        if save:
            plt.savefig(
                os.path.join(self.output_dir, "sediment_load_total.png"),
                bbox_inches="tight",
            )
        return plt.gca()

    def plot_sediment_load_fractions(
        self,
        scenario_index: int = 0,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        save: bool = False,
    ) -> None:
        """
        Plot sediment load fractions per phi class as stacked area for a scenario.

        Parameters
        ----------
        scenario_index : int, default=0
            Index of the scenario to plot
        start_date : str or datetime, optional
            Start date for the plot
        end_date : str or datetime, optional
            End date for the plot
        save : bool, default=False
            Whether to save the plot
        """
        scenario = self.reach.scenarios[scenario_index]
        if not hasattr(scenario, "sediment_load"):
            raise ValueError(f"Scenario {scenario.name} has no sediment_load data.")

        df = scenario.sediment_load.copy()
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        if start_date and end_date:
            mask = (df["Datetime"] >= start_date) & (df["Datetime"] <= end_date)
            df = df.loc[mask]

        phi_cols = [c for c in df.columns if c.startswith("qS_phi_")]
        plt.figure(figsize=(12, 6))
        plt.stackplot(df["Datetime"], df[phi_cols].T, labels=phi_cols, alpha=0.8)
        plt.xlabel("Date")
        plt.ylabel("Sediment Load per Phi Class (kg/s)")
        plt.title(f"{self.reach.name} - Sediment Load Fractions - {scenario.name}")
        plt.legend(loc="upper right", ncol=2)
        plt.grid(True)

        if save:
            plt.savefig(
                os.path.join(
                    self.output_dir, f"sediment_load_fractions_{scenario.name}.png"
                ),
                bbox_inches="tight",
            )
        return plt.gca()
