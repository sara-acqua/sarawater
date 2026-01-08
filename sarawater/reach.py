import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame

from sarawater.scenarios import Scenario, ConstScenario
from sarawater.IHA import compute_IHA


class Reach:
    def __init__(self, name: str, dates: list, Qnat: ndarray, Qab_max: float):
        """Represents a river reach.

        Parameters
        ----------
        name : str
            Name of the reach.
        dates : list[datetime]
            List of dates for the time series.
        Qnat : ndarray
            Natural flow rate time series.
        Qab_max : float
            Maximum value for the water abstraction.
        """
        if len(dates) != len(Qnat):
            raise ValueError("Dates and flow data length mismatch")

        self.name = name
        self.dates = dates
        self.Qnat = Qnat
        self.Qab_max = Qab_max
        self.scenarios: list[Scenario] = []
        self.IHA_nat = compute_IHA(Qnat, Qnat, dates)

    def __str__(self):
        return f"{self.name} is a Reach object with a flow time series with {len(self.Qnat)} elements. The date range starts from {min(self.dates)} and has {len(self.dates)} elements. The maximum flow abstraction is Qab_max={self.Qab_max} m3/s. So far, {len(self.scenarios)} scenarios have been added."

    def add_scenario(self, scenario: Scenario):
        """Add a scenario to the reach.

        Parameters
        ----------
        scenario : Scenario
            The scenario to add.

        Returns
        -------
        Reach
            The current reach instance.
        """
        self.scenarios.append(scenario)
        return self

    def print_scenarios(self):
        """Print the list of scenarios added to the reach."""
        for i, scenario in enumerate(self.scenarios):
            print(f"scenarios[{i}]: {scenario.name} | {scenario.description}")
        return None

    def add_ecological_flow_scenario(
        self, name: str, description: str, k: float = 0.2, p: float = 2.0
    ) -> Scenario:
        """Add an ecological flow scenario to the reach.

        Parameters
        ----------
        name : str
            Name of the scenario
        description : str
            Description of the scenario
        k : float, optional
            Protection factor, by default 0.2
        p : float, optional
            Nature protection factor, by default 2.0

        Returns
        -------
        Scenario
            The created ecological flow scenario
        """
        # Calculate monthly averages from daily data
        monthly_means = np.zeros(12)
        for month in range(1, 13):
            # Create mask for current month across all years
            month_mask = np.array([d.month == month for d in self.dates])
            if np.any(month_mask):  # Check if we have data for this month
                monthly_means[month - 1] = np.mean(self.Qnat[month_mask])
            else:
                raise ValueError(f"No data available for month {month}")

        # Calculate overall mean flow from daily data
        Q_mean = np.mean(self.Qnat)

        # Calculate Q97 from daily data (approximation of Q355)
        Q97 = np.percentile(self.Qnat, 3)

        # Calculate DE for each month
        QR_months = []
        for Q_month in monthly_means:
            M1 = np.sqrt(Q_month / Q_mean)
            DE = k * p * M1 * Q_mean
            QR_months.append(max(DE, Q97))

        # Create and return constant scenario with computed monthly values
        scenario = ConstScenario(name, description, self, QR_months)
        self.add_scenario(scenario)
        return scenario

    def add_HQ_curve(self, HQ_curve: DataFrame):
        """Add a habitat-flow curve to the reach.

        Parameters
        ----------
        HQ_curve : DataFrame
            Habitat-flow curve as a pandas DataFrame where columns represent different species/stages.
        """
        self.HQ_curve = HQ_curve
        self.HQ_curve_columns = list(HQ_curve.columns)
        # Exclude "DIS" and "WET" columns from available curves
        self.available_HQ_curves = [
            col for col in self.HQ_curve_columns if col not in ["DIS", "WET"]
        ]
        return HQ_curve

    def get_list_available_HQ_curves(self) -> list:
        """Get the list of available HQ curve names.

        Returns
        -------
        list
            List of column names representing available HQ curves.
        """
        if hasattr(self, "available_HQ_curves"):
            return self.available_HQ_curves
        else:
            return []

    def get_HQ_curve(self, curve_name: str) -> DataFrame:
        """Get a specific habitat-flow curve by name.

        Parameters
        ----------
        curve_name : str
            Name of the desired HQ curve (species).

        Returns
        -------
        DataFrame
            The requested HQ curve as a pandas DataFrame.
        """
        if hasattr(self, "HQ_curve"):
            if curve_name in self.HQ_curve_columns:
                HQ_Q = self.HQ_curve["DIS"]
                HQ_H = self.HQ_curve[curve_name]
                HQ = pd.DataFrame({"DIS": HQ_Q, curve_name: HQ_H})
                return HQ
            else:
                raise ValueError(f"HQ curve '{curve_name}' not found in the reach.")
        else:
            raise ValueError("No HQ curves have been added to this reach.")

    def add_cross_section_info(self, section, slope, grain_data):
        """Add cross-section information to the reach, including geometry, slope,
        and optionally grain size distribution.

        Parameters
        ----------
        section : str or DataFrame
            Path to CSV file containing section coordinates ('x [m]', 'y [m]'),
            or a pandas DataFrame with those columns.
        slope : float
            Channel bed slope (m/m).
        grain_data : str, DataFrame, float
            Path to CSV file or DataFrame. Must contain 'i(di)' and 'di[mm]' columns

        Returns
        -------
        Reach
            The current reach instance.
        """
        # --- Load section geometry ---
        if isinstance(section, str):
            section_data = pd.read_csv(section, delimiter=None, engine="python")
        elif isinstance(section, pd.DataFrame):
            section_data = section.copy()
        else:
            raise ValueError("section must be a CSV file path or a pandas DataFrame")

        # Validate columns
        required_cols = ["x [m]", "y [m]"]
        if not all(col in section_data.columns for col in required_cols):
            raise ValueError(f"Section data must contain columns {required_cols}")

        # --- Handle slope ---
        if not isinstance(slope, (float, int)):
            raise ValueError("slope must be a numeric value (float)")

        # Compute section width, area and equivalent rectangular geometry
        x = section_data["x [m]"].values
        y = section_data["y [m]"].values
        x_min, x_max = np.min(x), np.max(x)
        y_max = np.max(y)
        width = x_max - x_min
        area = np.trapezoid(y, x)
        height_avg = area / width

        num_points = len(section_data)
        x_equiv = np.linspace(x_min, x_max, num_points)
        y_equiv = np.full(num_points, height_avg)
        x_with_edges = np.concatenate(([x_min], x_equiv, [x_max]))
        y_with_edges = np.concatenate(([y_max], y_equiv, [y_max]))
        rectangular_section = pd.DataFrame(
            {"x [m]": x_with_edges, "y [m]": y_with_edges}
        )

        # --- Handle grain size data ---
        if grain_data is not None:
            # If a path or DataFrame was passed earlier it's already handled above.
            # Here we accept a float (D50), or a 2D array/list/ndarray with two columns:
            # [i(di), di[mm]] or shape (2, n) with first row i(di) and second row di[mm]).
            if isinstance(grain_data, (float, int)):
                # Treat as D50 in mm — construct a minimal, reasonable distribution
                d50 = float(grain_data)

                dfphi = pd.DataFrame(
                    {
                        "i(di)": [0.0, 1.0],  # cumulative fractions (0..1)
                        "di[mm]": [d50, d50],  # grain sizes in mm
                    }
                )
            elif isinstance(grain_data, (list, tuple, np.ndarray)):
                arr = np.asarray(grain_data)
                if arr.ndim == 2 and arr.shape[1] == 2:
                    dfphi = pd.DataFrame(arr, columns=["i(di)", "di[mm]"])
                elif arr.ndim == 2 and arr.shape[0] == 2:
                    # transposed form: first row i(di), second row di[mm]
                    dfphi = pd.DataFrame({"i(di)": arr[0, :], "di[mm]": arr[1, :]})
                else:
                    raise ValueError(
                        "grain_data array must be shape (n,2) or (2,n) with columns [i(di), di[mm]]"
                    )
            elif isinstance(grain_data, str):
                dfphi = pd.read_csv(grain_data)
            else:
                raise ValueError(
                    "grain_data must be float (D50), a 2D array/list with [i(di), di[mm]], a DataFrame, or a path to a CSV file"
                )

            # Ensure numeric columns and sensible ordering
            dfphi = dfphi.copy()
            if "i(di)" not in dfphi.columns or "di[mm]" not in dfphi.columns:
                raise ValueError("grain_data must provide columns 'i(di)' and 'di[mm]'")

            dfphi["i(di)"] = pd.to_numeric(dfphi["i(di)"], errors="coerce")
            dfphi["di[mm]"] = pd.to_numeric(dfphi["di[mm]"], errors="coerce")

            if dfphi["i(di)"].isnull().any() or dfphi["di[mm]"].isnull().any():
                raise ValueError("grain_data contains non-numeric values")

            # If i(di) appears to be percentages (0-100), convert to fractions (0-1)
            if dfphi["i(di)"].max() > 1.0:
                dfphi["i(di)"] = dfphi["i(di)"] / 100.0

            # Force cumulative behavior: sort by di and ensure monotonic cumulative values
            dfphi = dfphi.sort_values("di[mm]").reset_index(drop=True)
            # Clip cumulative to [0,1] and enforce non-decreasing
            dfphi["i(di)"] = dfphi["i(di)"].clip(0.0, 1.0)
            dfphi["i(di)"] = np.maximum.accumulate(dfphi["i(di)"])

            dfphi["di(Fehr) [mm]"] = dfphi["di[mm]"].interpolate()
            dfphi["Phi Scale"] = -np.log2(dfphi["di(Fehr) [mm]"])
            dfphi["Percent"] = (
                dfphi["i(di)"].diff().fillna(dfphi["i(di)"].iloc[0]) * 100
            )
            phi_classes = np.arange(-9.5, 7.5 + 1, 1)
            dfphi["Phi Interval"] = pd.cut(
                dfphi["Phi Scale"],
                bins=phi_classes,
                right=False,
                labels=phi_classes[:-1],
            )
            phi_percentages = dfphi.groupby("Phi Interval")["Percent"].sum()
            if 7.5 not in phi_percentages.index:
                phi_percentages.loc[7.5] = 0.0
        else:
            dfphi, phi_percentages = None, None

        # Store results
        self.section_data = section_data
        self.rectangular_section = rectangular_section
        self.width = width
        self.area = area
        self.height_avg = height_avg
        self.slope = slope
        self.grain_size_data = dfphi
        self.phi_percentages = phi_percentages

        return self

    def export_scenarios_summary(
        self, output_path: str = None, format: str = "csv"
    ) -> DataFrame:
        """Export a comprehensive summary table of all scenarios with their parameters and indices.

        This method generates a user-friendly table containing, for each scenario:
        - Scenario metadata (name, description)
        - Scenario parameters (QR values, Qab_max)
        - IHA/IARI indices (aggregated and by group)
        - Abstracted volumes (yearly totals, monthly averages)
        - Monthly released flows
        - Habitat indices (IH, ISH, ITH, HSD) if available
        - Annual sediment budget (total and per phi class) if available

        Parameters
        ----------
        output_path : str, optional
            Path where to save the export file. If None, only returns the DataFrame without saving.
        format : str, default='csv'
            Export format. Options: 'csv', 'excel'. Only used if output_path is provided.

        Returns
        -------
        DataFrame
            A pandas DataFrame containing the comprehensive summary of all scenarios.

        Raises
        ------
        ValueError
            If no scenarios have been added to the reach.
            If format is not 'csv' or 'excel'.

        Examples
        --------
        >>> reach = Reach("MyReach", dates, Qnat, Qab_max)
        >>> # ... add scenarios and compute their metrics ...
        >>> df = reach.export_scenarios_summary("output.csv", format="csv")
        >>> df = reach.export_scenarios_summary("output.xlsx", format="excel")
        """
        if len(self.scenarios) == 0:
            raise ValueError("No scenarios have been added to this reach.")

        if format not in ["csv", "excel"]:
            raise ValueError("format must be 'csv' or 'excel'")

        # Collect data for all scenarios
        data_rows = []

        for scenario in self.scenarios:
            row = {
                "scenario_name": scenario.name,
                "scenario_description": scenario.description,
            }

            # Add scenario parameters
            row["Qab_max"] = scenario.Qab_max

            # Add scenario-specific parameters
            if hasattr(scenario, "QR_months"):
                row["scenario parameters"] = scenario.QR_months
                # for i, qr in enumerate(scenario.QR_months):
                #     row[f"QR_month_{i+1}"] = qr
            elif hasattr(scenario, "QRbase"):
                row["scenario parameters"] = [
                    scenario.QRbase,
                    scenario.c_Qin,
                    scenario.QRmin,
                    scenario.QRmax,
                ]

            # Add monthly released flows (average per month)
            if scenario.QS is not None:
                months = np.array([d.month for d in scenario.dates])
                for month in range(1, 13):
                    month_mask = months == month
                    if np.any(month_mask):
                        row[f"QS_mean_month_{month}_m3s"] = np.mean(
                            scenario.QS[month_mask]
                        )

            # Add volume statistics if available
            if hasattr(scenario, "yearly_abs_volumes") and hasattr(
                scenario, "yearly_nat_volumes"
            ):
                row["yearly_abs_volume_mean_m3"] = np.mean(scenario.yearly_abs_volumes)
                row["yearly_nat_volume_mean_m3"] = np.mean(scenario.yearly_nat_volumes)

                # Normalized abstracted volume (handle division by zero)
                # Set to 0 where natural volume is 0 to avoid division errors
                with np.errstate(divide="ignore", invalid="ignore"):
                    abs_norm = np.where(
                        scenario.yearly_nat_volumes != 0,
                        scenario.yearly_abs_volumes / scenario.yearly_nat_volumes,
                        0.0,
                    )
                row["abs_volume_normalized_mean"] = np.mean(abs_norm)

            # Add monthly abstracted volumes if available
            if hasattr(scenario, "monthly_abs_volumes"):
                for i, vol in enumerate(scenario.monthly_abs_volumes):
                    row[f"monthly_abs_volume_month_{i+1}_m3"] = vol

            # Add seasonal volumes if available
            if hasattr(scenario, "seasonal_abs_volumes"):
                for season, vol in scenario.seasonal_abs_volumes.items():
                    row[f"seasonal_abs_volume_{season}_m3"] = vol

            # Add cases duration if available
            if hasattr(scenario, "cases_duration"):
                row["case1_duration_fraction"] = scenario.cases_duration[0]
                row["case2_duration_fraction"] = scenario.cases_duration[1]
                row["case3_duration_fraction"] = scenario.cases_duration[2]

            # Add habitat indices if available
            ih_dict = getattr(scenario, "IH", {})
            if ih_dict:
                for species, ih_data in ih_dict.items():
                    row[f"IH_{species}"] = ih_data.get("IH", np.nan)
                    row[f"ISH_{species}"] = ih_data.get("ISH", np.nan)
                    row[f"ITH_{species}"] = ih_data.get("ITH", np.nan)
                    row[f"HSD_{species}"] = ih_data.get("HSD", np.nan)

            # Add IHA indices if available (IARI)
            if hasattr(scenario, "IARI"):
                iari_dict = scenario.IARI
                row["IARI_aggregated_mean"] = np.mean(iari_dict["aggregated"])
                for group_name, group_values in iari_dict["groups"].items():
                    row[f"IARI_{group_name}_mean"] = np.mean(group_values)

            # Add normalized IHA indices if available
            if hasattr(scenario, "normalized_IHA"):
                norm_iha_dict = scenario.normalized_IHA
                row["normalized_IHA_aggregated_mean"] = np.mean(
                    norm_iha_dict["aggregated"]
                )
                for group_name, group_values in norm_iha_dict["groups"].items():
                    row[f"normalized_IHA_{group_name}_mean"] = np.mean(group_values)

            # Add annual sediment budget if available
            if hasattr(scenario, "annual_sediment_budget"):
                budget = scenario.annual_sediment_budget
                # If it's a DataFrame, compute mean values
                if isinstance(budget, pd.DataFrame):
                    # Add mean total sediment budget
                    if "qS_total" in budget.columns:
                        row["annual_sediment_budget_total_mean"] = budget[
                            "qS_total"
                        ].mean()

                # If it's a dict, compute mean from yearly values
                elif isinstance(budget, dict):
                    # Structure is {year: {phi_class: value, ...}}
                    all_years = list(budget.keys())
                    first_year_data = budget[all_years[0]]
                    if "qS_total" in first_year_data:
                        total_values = [budget[year]["qS_total"] for year in all_years]
                        row["annual_sediment_budget_total_mean"] = np.mean(total_values)

            data_rows.append(row)

        # Create DataFrame
        df = pd.DataFrame(data_rows)

        # Save to file if output_path is provided
        if output_path is not None:
            if format == "csv":
                df.to_csv(output_path, index=False)
            elif format == "excel":
                try:
                    df.to_excel(output_path, index=False, engine="openpyxl")
                except ImportError:
                    raise ImportError(
                        "Excel export requires the 'openpyxl' package. "
                        "Install it with: pip install openpyxl"
                    )

        return df
