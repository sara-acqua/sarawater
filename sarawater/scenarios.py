import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import ndarray

from sarawater.IHA import compute_IHA_index, compute_IHA
from sarawater import habitat as hab
from sarawater.sediment_load import (
    compute_sediment_load,
    compute_annual_sediment_volume,
)


class Scenario:
    def __init__(self, name: str, description: str, reach: "Reach", Qab_max=None):  # type: ignore
        """Parent class for all types of scenarios. Contains the name and description of the scenario.

        Parameters
        ----------
        name : str
            Name of the scenario.
        description : str
            Description of the scenario.
        reach : Reach
            The reach object associated with this scenario.
        Qab_max : float, optional
            Maximum value for the water abstraction, by default None (which takes the value from the reach object).
        """
        self.name = name
        self.description = description
        self.reach = reach
        self.QR = None  # Placeholder for the minimum release flow time series
        self.QS = None  # Placeholder for the released flow rate time series
        if Qab_max is None:
            self.Qab_max = reach.Qab_max
        else:
            self.Qab_max = Qab_max
        self.IH = {}
        self.IHA = None  # Placeholder for IHA indicators

    def __repr__(self):
        return f"Scenario(name={self.name}, description={self.description}, reach={self.reach.name})"

    @property
    def Qnat(self) -> ndarray:
        """Get the natural flow rate time series from the associated Reach."""
        return self.reach.Qnat

    @property
    def dates(self) -> list:
        """Get the dates from the associated Reach."""
        return self.reach.dates

    def compute_QS(self) -> ndarray:
        """Compute the released flow rate time series for the scenario.

        Returns
        -------
        ndarray
            Released flow rate time series.
        """
        if self.QR is None:
            raise ValueError("QR must be set before computing QS.")

        QS = np.zeros_like(self.Qnat)
        case1 = self.Qnat <= self.QR
        case2 = (self.Qnat > self.QR) & (self.Qnat < self.Qab_max + self.QR)
        case3 = self.Qnat >= self.Qab_max + self.QR
        QS[case1] = self.Qnat[case1]
        QS[case2] = self.QR[case2]
        QS[case3] = self.Qnat[case3] - self.Qab_max
        self.QS = QS
        self.cases_duration = [sum(c) / QS.size for c in [case1, case2, case3]]
        return QS

    def plot_scenario_discharge(
        self, start_date=None, end_date=None, **kwargs
    ) -> plt.Axes:
        """Plot released discharge (QS) for a given scenario within a specified date range.

        Parameters
        ----------
        start_date : str or datetime, optional
            Start date in format 'YYYY-MM-DD' or datetime object
        end_date : str or datetime, optional
            End date in format 'YYYY-MM-DD' or datetime object
        **kwargs : dict
            Additional keyword arguments to pass to matplotlib.pyplot.plot

        Returns
        -------
        plt.Axes
            The current Axes instance
        """
        # Convert string dates to datetime if provided
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)

        # Create date mask
        if start_date is not None and end_date is not None:
            mask = [(dt >= start_date) and (dt <= end_date) for dt in self.dates]
        else:
            mask = [True] * len(self.dates)

        # If label is not provided in kwargs, use scenario name
        if "label" not in kwargs:
            kwargs["label"] = self.name

        # Plot the data with any additional keyword arguments
        plt.plot(np.array(self.dates)[mask], self.QS[mask], **kwargs)

        # Customize the plot
        plt.title(f"Flow release time series for {self.reach.name}")
        plt.xlabel("Date")
        plt.ylabel("Discharge (m³/s)")
        plt.grid(True)

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)

        # Adjust layout to prevent label cutoff
        plt.tight_layout()

        return plt.gca()

    def compute_IHA(self, **kwargs) -> dict:
        """Compute the IHA for the scenario using the function compute_IHA().
        See the function documentation in IHA.py for more details on parameters and return values.

        Returns
        -------
        dict[str, dict[str, np.ndarray]]
            Dictionary containing IHA indicators grouped by type:
            {
                'Group1': {
                    'mean_january': np.array([yearly values]),
                    ...
                },
                ...
                'Group5': {...}
            }
        """
        self.IHA = compute_IHA(self.Qnat, self.QS, self.dates, **kwargs)
        return self.IHA

    def compute_IHA_index(
        self,
        index_metric,
        index_options={},
    ) -> tuple[dict, dict]:
        """Compute the IHA index for the scenario using compute_IHA_index(). Currently supports 'IARI' and 'normalized_IHA' as index_metric.

        Parameters
        ----------
        index_metric : str
            Name of the index to compute ('IARI' or 'normalized_IHA')
        index_options : dict, optional
            Keyword arguments to pass to the compute_IHA_index function.
            See the function documentation in IHA.py for more details.

        Returns
        -------
        tuple[dict, dict]
            A tuple containing:
            1. Dictionary containing IHA indicators grouped by type for the altered state
            2. Dictionary containing the index values (IARI or normalized_IHA) per group and aggregated
        """
        if self.IHA is None:
            self.compute_IHA()

        _, out_dict = compute_IHA_index(
            self.Qnat,
            self.QS,
            self.dates,
            index_metric=index_metric,
            IHA_nat=self.reach.IHA_nat,
            IHA_alt=self.IHA,
            **index_options,
        )
        if index_metric.lower() == "iari":
            self.IARI = out_dict
        elif index_metric.lower() == "normalized_iha":
            self.normalized_IHA = out_dict
        else:
            raise ValueError("index_metric must be either 'IARI' or 'normalized_IHA'")
        return self.IHA, out_dict

    def compute_natural_abstracted_volumes(
        self,
        month_to_season: dict[int, str] = None,
    ) -> tuple[ndarray, ndarray, ndarray, ndarray]:
        """Compute the water volumes abstracted for the scenario.
        Returns yearly totals and monthly averages over the whole series.

        Parameters
        ----------
        month_to_season : dict[int, str], optional
            Dictionary mapping month numbers (1-12) to season names.
            If not provided, uses default mapping:
            - Winter: Dec, Jan, Feb, Mar
            - Spring: Apr, May, Jun, Jul
            - Summer: Aug, Sep
            - Autumn: Oct, Nov

        Returns
        -------
        tuple[ndarray, ndarray, ndarray, ndarray]
            - Natural volumes per year
            - Abstracted volumes per year
            - Average natural volumes per month
            - Average abstracted volumes per month
        """
        if self.QS is None:
            raise ValueError(
                "QS must be computed before calculating abstracted volumes."
            )

        # Default season mapping if none provided
        if month_to_season is None:
            month_to_season = {
                12: "Winter",
                1: "Winter",
                2: "Winter",
                3: "Winter",
                4: "Spring",
                5: "Spring",
                6: "Spring",
                7: "Spring",
                8: "Summer",
                9: "Summer",
                10: "Autumn",
                11: "Autumn",
            }

        # Get time step in seconds from consecutive dates
        time_steps = np.array(
            [
                (self.dates[i + 1] - self.dates[i]).total_seconds()
                for i in range(len(self.dates) - 1)
            ]
        )
        # Add the last time step (assume same as previous)
        time_steps = np.append(time_steps, time_steps[-1])

        # Calculate abstracted flow rates
        Qab = self.Qnat - self.QS

        # Convert flow rates (m³/s) to volumes (m³)
        nat_volumes = self.Qnat * time_steps
        abs_volumes = Qab * time_steps

        # Calculate yearly totals
        years = np.array([d.year for d in self.dates])
        yearly_nat = np.array([nat_volumes[years == y].sum() for y in np.unique(years)])
        yearly_abs = np.array([abs_volumes[years == y].sum() for y in np.unique(years)])

        # Calculate monthly averages
        months = np.array([d.month for d in self.dates])
        monthly_nat = np.array(
            [np.mean(nat_volumes[months == m]) for m in range(1, 13)]
        )
        monthly_abs = np.array(
            [np.mean(abs_volumes[months == m]) for m in range(1, 13)]
        )

        # Calculate seasonal volumes
        seasons = set(month_to_season.values())
        seasonal_nat = {season: 0.0 for season in seasons}
        seasonal_abs = {season: 0.0 for season in seasons}

        for month in range(1, 13):
            season = month_to_season[month]
            seasonal_nat[season] += monthly_nat[month - 1]
            seasonal_abs[season] += monthly_abs[month - 1]

        # Store as instance attributes
        self.yearly_nat_volumes = yearly_nat
        self.yearly_abs_volumes = yearly_abs
        self.monthly_nat_volumes = monthly_nat
        self.monthly_abs_volumes = monthly_abs
        self.seasonal_nat_volumes = seasonal_nat
        self.seasonal_abs_volumes = seasonal_abs

        return yearly_nat, yearly_abs, monthly_nat, monthly_abs

    def cases_duration_for_month(self, month: int) -> list[float]:
        """
        Compute the duration percentage of each flow case for a specific month.

        Parameters
        ----------
        month : int
            Month to filter by (1-12).

        Returns
        -------
        list[float]
            List with the duration percentage of each case [case1, case2, case3] for the specified month.
        """
        if self.QS is None or self.QR is None:
            raise ValueError(
                "QS and QR must be computed before calculating case durations."
            )
        month_mask = np.array([date.month == month for date in self.dates])
        Qnat_m = self.Qnat[month_mask]
        QR_m = self.QR[month_mask]
        # Compute cases for the month
        case1 = Qnat_m <= QR_m
        case2 = (Qnat_m > QR_m) & (Qnat_m < QR_m + self.Qab_max)
        case3 = Qnat_m >= QR_m + self.Qab_max
        total = len(Qnat_m)
        if total == 0:
            return [0, 0, 0]
        return [np.sum(case1) / total, np.sum(case2) / total, np.sum(case3) / total]

    def compute_IH_for_species(
        self, species: str | list[str] | None = None, **kwargs
    ) -> dict:
        """Compute the Habitat Index (IH) for a given species using the scenario's QS.

        Parameters
        ----------
        species : str, list[str], or None, optional
            Target species for which to compute the Habitat Index.
            If None (default), computes IH for all available species.
            If str, computes IH for a single species.
            If list[str], computes IH for the specified list of species.

        Returns
        -------
        dict
            Dictionary with results from "compute_habitat_indices" for each species.
        """
        if self.QS is None:
            raise ValueError("QS must be computed before calculating IH.")

        # Determine which species to process
        if species is None:
            species_list = self.reach.get_list_available_HQ_curves()
        elif isinstance(species, str):
            species_list = [species]
        else:
            species_list = species

        # Compute IH for each species
        for sp in species_list:
            HQ = self.reach.get_HQ_curve(sp)
            IH_values = hab.compute_habitat_indices(
                self.Qnat, self.QS, HQ, self.dates, **kwargs
            )
            self.IH[sp] = IH_values

        # Return the last computed values for backwards compatibility
        return self.IH

    def compute_sediment_load_from_reach(self, to_csv=None):
        """
        Compute the sediment load time series for the current reach using stored discharge and reach properties.
        This method wraps the lower-level compute_sediment_load function by pulling the required
        parameters from the object's stored state (discharge series and reach geometry/grain-size
        information) and delegating the actual computation.

        Parameters
        ----------
        to_csv : optional, str or file-like or bool, default=None
            If a path-like string or file-like object is provided, the computed results will be
            written to CSV at that location or using that file handle. If None or False, no CSV
            file is written. The exact accepted values are forwarded to compute_sediment_load.

        Returns
        -------
        object
            The return value from compute_sediment_load (typically a time series or table of
            sediment loads indexed by the provided datetime_series). The concrete type and
            structure depend on the implementation of compute_sediment_load.

        Raises
        ------
        ValueError
            If self.QS is None (discharge must be computed before calling this method).
        """
        if self.QS is None:
            raise ValueError("QS must be computed first.")

        try:
            B = self.reach.width
            slope = self.reach.slope
            Fi = self.reach.phi_percentages.values
        except AttributeError:
            raise ValueError(
                "Reach properties (width, slope, grain size distribution) must be set before computing sediment load. Run Reach.add_cross_section_info() first."
            )

        return compute_sediment_load(
            self.QS,
            self.dates,
            B,
            slope,
            Fi,
            to_csv=to_csv,
        )

    def plot_scenario_sediment_transport(
        self, start_date=None, end_date=None, unit="m3_per_day", rho_s=2650, **kwargs
    ) -> plt.Axes:
        """Plot sediment transport capacity for a given scenario within a specified date range.

        Parameters
        ----------
        start_date : str or datetime, optional
            Start date in format 'YYYY-MM-DD' or datetime object
        end_date : str or datetime, optional
            End date in format 'YYYY-MM-DD' or datetime object
        unit : str, optional
            Unit for sediment transport: 'm3_per_day' (default), 'm3_per_s', or 'ton_per_day'
        rho_s : float, optional
            Sediment density in kg/m³ used when converting to mass (ton/day). Default 2650.
        **kwargs : dict
            Additional keyword arguments to pass to matplotlib.pyplot.plot

        Returns
        -------
        plt.Axes
            The current Axes instance
        """
        # Check if sediment transport has been computed
        if not hasattr(self, "sediment_load_df") or self.sediment_load_df is None:
            # Compute it if not available
            self.sediment_load_df = self.compute_sediment_load_from_reach()

        # Convert string dates to datetime if provided
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)

        # Create date mask
        if start_date is not None and end_date is not None:
            mask = [(dt >= start_date) and (dt <= end_date) for dt in self.dates]
        else:
            mask = [True] * len(self.dates)
        mask = np.array(mask, dtype=bool)

        # Extract sediment transport data and convert units
        qS_total = np.asarray(self.sediment_load_df["qS_total"].values)

        if unit == "m3_per_day":
            qS_plot = qS_total * 86400  # Convert m³/s to m³/day
            ylabel = "Sediment transport [m³/day]"
        elif unit in ("m3_per_s", "m3/s", "m3_per_second"):
            qS_plot = qS_total
            ylabel = "Sediment transport [m³/s]"
        elif unit == "ton_per_day":
            # Convert m³/s -> m³/day -> kg/day (rho_s kg/m³) -> ton/day (divide by 1000)
            qS_plot = qS_total * 86400 * rho_s / 1000.0
            ylabel = "Sediment transport [ton/day]"
        else:
            raise ValueError(
                "Unknown unit '{}'. Supported units: 'm3_per_day', 'm3_per_s' (or 'm3/s'), 'ton_per_day'.".format(
                    unit
                )
            )

        # If label is not provided in kwargs, use scenario name
        if "label" not in kwargs:
            kwargs["label"] = self.name

        # Plot the data with any additional keyword arguments
        plt.plot(np.array(self.dates)[mask], qS_plot[mask], **kwargs)

        # Customize the plot
        plt.title(f"Sediment transport capacity for {self.reach.name}")
        plt.xlabel("Date")
        plt.ylabel(ylabel)
        plt.grid(True)

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)

        # Adjust layout to prevent label cutoff
        plt.tight_layout()

        return plt.gca()

    def compute_annual_sediment_budget(
        self, to_ton=False, rho_s=2650, to_csv=None, as_dict=False
    ):
        """
        Compute annual sediment volume or mass (ton/year) from the scenario's sediment load.

        Parameters
        ----------
        to_ton : bool, optional
            If True, convert m³ to ton using rho_s.
        rho_s : float, optional
            Sediment density in kg/m³. Default is 2650 kg/m³.
        to_csv : str, optional
            File path to save the annual sediment budget table.
        as_dict : bool, optional
            If True, return a dictionary instead of a DataFrame.

        Returns
        -------
        pd.DataFrame or dict
            Annual sediment budget (per phi class and total) in m³/year or ton/year.
        """
        # First compute the sediment load time series
        sed_df = self.compute_sediment_load_from_reach(to_csv=to_csv)
        self.sediment_load_df = sed_df

        # Then compute annual volumes or tons
        annual_budget = compute_annual_sediment_volume(
            sed_df, to_csv=to_csv, as_dict=as_dict, to_ton=to_ton, rho_s=rho_s
        )
        self.annual_sediment_budget = annual_budget
        return annual_budget


class ConstScenario(Scenario):
    def __init__(
        self, name: str, description: str, reach: "Reach", QR_months: list[float], **kwargs  # type: ignore
    ):
        """Constant flow rate scenario.

        Parameters
        ----------
        name : str
            Name of the scenario.
        description : str
            Description of the scenario.
        reach : Reach
            The reach object associated with this scenario.
        QR_months : list[float]
            Monthly constant flow rates.
        """
        super().__init__(name, description, reach, **kwargs)

        if len(QR_months) != 12:
            raise ValueError("QR_months must have 12 elements.")
        self.QR_months = QR_months

        # Map the monthly flow rates to the dates of the reach
        self.QR = np.zeros_like(self.Qnat)
        for i, month in enumerate(self.QR_months):
            month_mask = np.array([date.month == i + 1 for date in self.dates])
            self.QR[month_mask] = month


class PropScenario(Scenario):
    def __init__(
        self,
        name: str,
        description: str,
        reach: "Reach",  # type: ignore
        QRbase: float,
        c_Qin: float,
        QRmin: float,
        QRmax: float,
        **kwargs,
    ):
        """Proportional flow rate scenario.

        Parameters
        ----------
        name : str
            Name of the scenario.
        description : str
            Description of the scenario.
        reach : Reach
            The reach object associated with this scenario.
        QRbase : float
            Base flow rate.
        c_Qin : float
            Coefficient for inflow.
        QRmin : float
            Minimum value for the prescribed minimum released flow rate.
        QRmax : float
            Maximum value for the prescribed minimum released flow rate.
        """
        super().__init__(name, description, reach, **kwargs)
        self.QRbase = QRbase
        self.c_Qin = c_Qin
        self.QRmin = QRmin
        self.QRmax = QRmax
        self.QR = QRbase + c_Qin * self.Qnat
        self.QR[self.QR < QRmin] = QRmin
        self.QR[self.QR > QRmax] = QRmax
