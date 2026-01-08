import numpy as np
import datetime
import pandas as pd

from sarawater.utils import compute_consecutive_lengths


def compute_IHA(
    Qnat: np.ndarray, QS: np.ndarray, dates: list, zero_flow_threshold: float = 0.001
) -> dict[str, dict[str, np.ndarray]]:
    """Compute Indicators of Hydrologic Alteration (IHA) for a given flow time series QS with respect to a natural time series Qnat.
    Each indicator is computed yearly. The indicators are grouped into 5 groups as per IHA methodology.

    Note: If the input data has sub-daily resolution (e.g., hourly), it will be automatically aggregated to daily averages
    before computing IHA parameters, as IHA methodology requires daily time series data.

    Parameters
    ----------
    Qnat : np.ndarray
        Natural flow rate time series (any temporal resolution; will be aggregated to daily)
    QS : np.ndarray
        Released flow rate time series (any temporal resolution; will be aggregated to daily)
    dates : list
        List of datetime objects corresponding to flow rates
    zero_flow_threshold : float, optional
        Threshold below which flow is considered zero in the "zero-flow days" indicator (default is 0.001)

    Returns
    -------
    dict[str, dict[str, np.ndarray]]
        Dictionary containing IHA indicators grouped by type
        {
            'Group1': {
                'mean_january': np.array([yearly values]),
                ...
                'mean_december': np.array([yearly values])
            },
            'Group2': {
                'base_flow': np.array([yearly values]),
                'moving_avg_1d_min': np.array([yearly values]),
                ...
            },
            ...
            'Group5': {...}
        }
    """
    # Force daily averaging of flow discharge data
    # Convert dates to date-only (removing time component)

    dates_array = np.array(dates)
    df = pd.DataFrame(
        {"date": pd.to_datetime(dates_array).date, "Qnat": Qnat, "QS": QS}
    )

    # Group by date and compute daily averages
    df_daily = df.groupby("date").agg({"Qnat": "mean", "QS": "mean"}).reset_index()

    # Extract daily-averaged data
    dates_daily = pd.to_datetime(df_daily["date"]).tolist()
    Qnat_daily = df_daily["Qnat"].values
    QS_daily = df_daily["QS"].values

    IHA_groups = {f"Group{i+1}": {} for i in range(5)}
    years = np.unique([d.year for d in dates_daily])
    n_years = len(years)

    # Group 1: Monthly statistics
    for month in range(1, 13):
        month_name = datetime.datetime(2000, month, 1).strftime("%B").lower()
        yearly_means = np.zeros(n_years)
        for i, year in enumerate(years):
            year_month_mask = np.array(
                [d.month == month and d.year == year for d in dates_daily]
            )
            if np.any(year_month_mask):
                yearly_means[i] = np.mean(QS_daily[year_month_mask])
        IHA_groups["Group1"][f"mean_{month_name}"] = yearly_means

    # Group 2: Moving averages, base flow, zero-flow days
    yearly_moving_avgs = {
        window: {"min": np.zeros(n_years), "max": np.zeros(n_years)}
        for window in [1, 3, 7, 30, 90]
    }
    yearly_base_flow = np.zeros(n_years)
    yearly_zero_flow_days = np.zeros(n_years)

    for i, year in enumerate(years):
        year_mask = np.array([d.year == year for d in dates_daily])
        year_data = QS_daily[year_mask]
        yearly_base_flow[i] = np.mean(year_data)
        yearly_zero_flow_days[i] = np.sum(year_data < zero_flow_threshold)

        for window in [1, 3, 7, 30, 90]:
            moving_avg = np.convolve(year_data, np.ones(window) / window, mode="valid")
            yearly_moving_avgs[window]["min"][i] = np.min(moving_avg)
            yearly_moving_avgs[window]["max"][i] = np.max(moving_avg)

    IHA_groups["Group2"]["base_flow"] = yearly_base_flow
    IHA_groups["Group2"]["zero_flow_days"] = yearly_zero_flow_days
    for window in [1, 3, 7, 30, 90]:
        IHA_groups["Group2"][f"moving_avg_{window}d_min"] = yearly_moving_avgs[window][
            "min"
        ]
        IHA_groups["Group2"][f"moving_avg_{window}d_max"] = yearly_moving_avgs[window][
            "max"
        ]

    # Group 3: Timing of annual extremes
    julian_days_max = np.zeros(n_years)
    julian_days_min = np.zeros(n_years)
    for i, year in enumerate(years):
        year_mask = np.array([d.year == year for d in dates_daily])
        year_data = QS_daily[year_mask]
        year_dates = np.array(dates_daily)[year_mask]

        max_idx = np.argmax(year_data)
        min_idx = np.argmin(year_data)

        julian_days_max[i] = year_dates[max_idx].timetuple().tm_yday
        julian_days_min[i] = year_dates[min_idx].timetuple().tm_yday

    IHA_groups["Group3"]["julian_day_max"] = julian_days_max
    IHA_groups["Group3"]["julian_day_min"] = julian_days_min

    # Group 4: Pulse analysis
    yearly_pulse_stats = {
        "low_count": np.zeros(n_years),
        "high_count": np.zeros(n_years),
        "low_duration": np.zeros(n_years),
        "high_duration": np.zeros(n_years),
    }

    for i, year in enumerate(years):
        year_mask = np.array([d.year == year for d in dates_daily])
        year_data = QS_daily[year_mask]
        year_nat = Qnat_daily[year_mask]

        low_limit = np.percentile(year_nat, 25)
        high_limit = np.percentile(year_nat, 75)

        low_pulse = year_data < low_limit
        high_pulse = year_data > high_limit

        low_lengths = compute_consecutive_lengths(low_pulse)
        high_lengths = compute_consecutive_lengths(high_pulse)

        yearly_pulse_stats["low_count"][i] = len(low_lengths)
        yearly_pulse_stats["high_count"][i] = len(high_lengths)
        yearly_pulse_stats["low_duration"][i] = (
            np.median(low_lengths) if len(low_lengths) > 0 else 0
        )
        yearly_pulse_stats["high_duration"][i] = (
            np.median(high_lengths) if len(high_lengths) > 0 else 0
        )

    IHA_groups["Group4"].update(
        {
            "low_pulse_count": yearly_pulse_stats["low_count"],
            "high_pulse_count": yearly_pulse_stats["high_count"],
            "low_pulse_duration": yearly_pulse_stats["low_duration"],
            "high_pulse_duration": yearly_pulse_stats["high_duration"],
        }
    )

    # Group 5: Flow variations
    yearly_variations = {
        "pos_med": np.zeros(n_years),
        "neg_med": np.zeros(n_years),
        "reversals": np.zeros(n_years),
    }

    for i, year in enumerate(years):
        year_mask = np.array([d.year == year for d in dates_daily])
        year_data = QS_daily[year_mask]
        flow_changes = np.diff(year_data)

        pos_changes = flow_changes[flow_changes >= 0]
        neg_changes = flow_changes[flow_changes <= 0]

        yearly_variations["pos_med"][i] = (
            np.median(pos_changes) if len(pos_changes) > 0 else 0
        )
        yearly_variations["neg_med"][i] = (
            np.median(neg_changes) if len(neg_changes) > 0 else 0
        )
        yearly_variations["reversals"][i] = np.sum(np.diff(np.signbit(flow_changes)))

        IHA_groups["Group5"].update(
            {
                "positive_variation_median": yearly_variations["pos_med"],
                "negative_variation_median": yearly_variations["neg_med"],
                "flow_reversals": yearly_variations["reversals"],
            }
        )

    return IHA_groups


def compute_IHA_index(
    Qnat: np.ndarray,
    QS: np.ndarray,
    dates: list,
    index_metric: str,
    weights: list[float] = None,
    IHA_nat: dict = None,
    IHA_alt: dict = None,
    epsilon: float = 1e-5,
) -> tuple[dict, dict[str, dict[str, np.ndarray]]]:
    """Compute the IHA indicators and the related IARI index for each year.

    Parameters
    ----------
    Qnat : np.ndarray
        Natural flow rate time series
    QS : np.ndarray
        Released flow rate time series
    dates : list
        List of datetime objects corresponding to flow rates
    index_metric : str
        Name of the index to compute (IARI, normalized_IHA)
    weights : list[float], optional
        List of 5 weights for each group of IHA parameters. Must sum to 1.
        If None, equal weights (0.2) will be used.
    IHA_nat : dict, optional
        Pre-computed IHA for the natural flow series. If provided, it will be used instead of computing it again.
    IHA_alt : dict, optional
        Pre-computed IHA for the altered flow series. If provided, it will be used instead of computing it again.
    epsilon : float, optional
        Small value to prevent division by zero in calculations. Used only if index_metric is "normalized_IHA". Default is 1e-5.

    Returns
    -------
    tuple[dict, dict[str, dict[str, np.ndarray]]]
        A tuple containing:
        1. Dictionary containing IHA_indexes per group and aggregated:
           {
               'groups': {
                   'Group1': np.array([yearly values]),
                   ...
                   'Group5': np.array([yearly values])
               },
               'aggregated': np.array([yearly values])
           }
        2. Dictionary containing IHA indicators grouped by type for the altered state:
           {
               'Group1': {
                   'mean_january': np.array([yearly values]),
                   ...
               },
               ...
               'Group5': {...}
           }
    """
    # Calculate IHA indicators for both series
    if IHA_nat is None:
        IHA_nat = compute_IHA(Qnat, Qnat, dates)  # Natural state
    if IHA_alt is None:
        IHA_alt = compute_IHA(Qnat, QS, dates)  # Altered state

    years = np.unique([d.year for d in dates])
    n_years = len(years)

    # Set default weights if none provided
    if weights is None:
        weights = [0.2] * 5
    elif len(weights) != 5:
        raise ValueError("weights must be a list of 5 values")
    elif abs(sum(weights) - 1.0) > 1e-10:  # Allow for floating point precision
        raise ValueError("weights must sum to 1")

    # Define weights for each group
    group_weights = {group_name: w for group_name, w in zip(IHA_nat.keys(), weights)}

    # define if we are computing IARI or normalized IHA
    if index_metric.lower() not in ["iari", "normalized_iha"]:
        raise ValueError("index_metric must be either 'IARI' or 'normalized_IHA'")
    elif index_metric.lower() == "normalized_iha":
        # Initialize group normalized IHA arrays
        normalized_IHA_groups = {
            group_name: np.zeros(n_years) for group_name in IHA_nat.keys()
        }

        # Compute normalized IHA for each indicator in each group
        for group_name in IHA_nat.keys():
            nat_group = IHA_nat[group_name]
            alt_group = IHA_alt[group_name]

            for indicator in nat_group.keys():
                nat_values = np.array(nat_group[indicator])
                alt_values = np.array(alt_group[indicator])

                # Check if there are any near-zero values in nat_values
                if np.any(np.abs(nat_values) < epsilon):
                    print(
                        f"Warning: Near-zero values detected in natural flow series for indicator '{indicator}'. Using alternative calculation to avoid division by zero."
                    )
                    p_ik = np.abs((alt_values + 1) / (nat_values + 1) - 1)
                else:
                    p_ik = np.abs(alt_values / nat_values - 1)

                # Add to group normalized IHA (mean of indicators)
                normalized_IHA_groups[group_name] += p_ik / len(nat_group)

        # Compute aggregated normalized IHA using provided weights
        normalized_IHA_aggregated = np.zeros(n_years)
        for group_name, w in group_weights.items():
            normalized_IHA_aggregated += w * normalized_IHA_groups[group_name]
        nIHA_dict = {
            "groups": normalized_IHA_groups,
            "aggregated": normalized_IHA_aggregated,
        }

        return IHA_alt, nIHA_dict

    elif index_metric.lower() == "iari":
        # Initialize group IARI arrays
        IARI_groups = {group_name: np.zeros(n_years) for group_name in IHA_nat.keys()}

        # Compute p_i,k for each indicator in each group
        for group_name in IHA_nat.keys():
            nat_group = IHA_nat[group_name]
            alt_group = IHA_alt[group_name]

            for indicator in nat_group.keys():
                nat_values = np.array(nat_group[indicator])
                alt_values = np.array(alt_group[indicator])

                # Calculate quartiles from natural series
                Q25 = np.percentile(nat_values, 25)
                Q75 = np.percentile(nat_values, 75)

                if Q75 == Q25:
                    print(
                        f"Warning: The first and third quartile for {indicator} in the natural flow series are equal to each other. The associated IARI parameter will be set to 0."
                    )

                # Calculate p_i,k according to the formula
                p_ik = np.zeros_like(alt_values)
                for k, X_ik in enumerate(alt_values):
                    if Q25 <= X_ik <= Q75 or Q75 == Q25:
                        p_ik[k] = 0
                    else:
                        p_ik[k] = min(
                            abs((X_ik - Q25) / (Q75 - Q25)),
                            abs((X_ik - Q75) / (Q75 - Q25)),
                        )

                # Add to group IARI (mean of indicators)
                IARI_groups[group_name] += p_ik / len(nat_group)

        # Compute aggregated IARI using provided weights
        IARI_aggregated = np.zeros(n_years)
        for group_name, w in group_weights.items():
            IARI_aggregated += w * IARI_groups[group_name]
        IARI_dict = {
            "groups": IARI_groups,
            "aggregated": IARI_aggregated,
        }

        return IHA_alt, IARI_dict
