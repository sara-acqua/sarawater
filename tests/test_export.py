import sys
import os
import numpy as np
import pandas as pd
import datetime
import tempfile

sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), "..")))

import sarawater.scenarios as sc
import sarawater.reach as rch


def create_test_reach():
    """Create a test reach with sample data"""
    dates = [
        datetime.datetime(2020, 1, 1) + datetime.timedelta(days=i) for i in range(365)
    ]
    Qnat = np.random.uniform(10, 100, len(dates))
    reach = rch.Reach("Test Reach", dates, Qnat, 30.0)
    return reach


def test_export_scenarios_summary_no_scenarios():
    """Test that export raises error when no scenarios exist"""
    reach = create_test_reach()

    try:
        reach.export_scenarios_summary()
        assert False, "Should raise ValueError when no scenarios exist"
    except ValueError as e:
        assert "No scenarios" in str(e)


def test_export_scenarios_summary_basic():
    """Test basic export functionality with a simple scenario"""
    reach = create_test_reach()

    # Add a constant scenario
    QR_months = [5.0] * 12
    scenario = sc.ConstScenario("Test Scenario", "A test scenario", reach, QR_months)
    reach.add_scenario(scenario)

    # Compute QS
    scenario.compute_QS()

    # Export without saving
    df = reach.export_scenarios_summary()

    # Verify basic structure
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1  # One scenario
    assert df.loc[0, "scenario_name"] == "Test Scenario"
    assert df.loc[0, "scenario_description"] == "A test scenario"
    assert df.loc[0, "Qab_max"] == 30.0


def test_export_scenarios_summary_with_prop_params():
    """Test export with proportional scenario parameters"""
    reach = create_test_reach()

    scenario = sc.PropScenario(
        "Prop Test",
        "Proportional scenario",
        reach,
        QRbase=5.0,
        c_Qin=0.2,
        QRmin=3.0,
        QRmax=20.0,
    )
    reach.add_scenario(scenario)
    scenario.compute_QS()

    df = reach.export_scenarios_summary()

    # Check proportional parameters are in the scenario parameters column
    assert "scenario parameters" in df.columns
    assert isinstance(df.loc[0, "scenario parameters"], list)
    assert df.loc[0, "scenario parameters"] == [5.0, 0.2, 3.0, 20.0]


def test_export_scenarios_summary_with_volumes():
    """Test export with volume calculations"""
    reach = create_test_reach()

    scenario = sc.ConstScenario("Vol Test", "Volume test", reach, [5.0] * 12)
    reach.add_scenario(scenario)
    scenario.compute_QS()
    scenario.compute_natural_abstracted_volumes()

    df = reach.export_scenarios_summary()

    # Check volume fields are present
    assert "yearly_abs_volume_mean_m3" in df.columns
    assert "yearly_nat_volume_mean_m3" in df.columns
    assert "abs_volume_normalized_mean" in df.columns

    # Check monthly volumes
    for i in range(12):
        col_name = f"monthly_abs_volume_month_{i+1}_m3"
        assert col_name in df.columns


def test_export_scenarios_summary_with_iari():
    """Test export with IARI indices"""
    reach = create_test_reach()

    scenario = sc.ConstScenario("IARI Test", "IARI test", reach, [5.0] * 12)
    reach.add_scenario(scenario)
    scenario.compute_QS()
    scenario.compute_IHA_index(index_metric="IARI")

    df = reach.export_scenarios_summary()

    # Check IARI fields are present
    assert "IARI_aggregated_mean" in df.columns
    assert not pd.isna(df.loc[0, "IARI_aggregated_mean"])

    # Check group IARI values
    for i in range(1, 6):
        assert f"IARI_Group{i}_mean" in df.columns
        assert not pd.isna(df.loc[0, f"IARI_Group{i}_mean"])


def test_export_scenarios_summary_with_normalized_iha():
    """Test export with normalized IHA indices"""
    reach = create_test_reach()

    scenario = sc.ConstScenario("nIHA Test", "nIHA test", reach, [5.0] * 12)
    reach.add_scenario(scenario)
    scenario.compute_QS()
    scenario.compute_IHA_index(index_metric="normalized_IHA")

    df = reach.export_scenarios_summary()

    # Check normalized IHA fields are present
    assert "normalized_IHA_aggregated_mean" in df.columns
    assert not pd.isna(df.loc[0, "normalized_IHA_aggregated_mean"])

    # Check group normalized IHA values
    for i in range(1, 6):
        assert f"normalized_IHA_Group{i}_mean" in df.columns
        assert not pd.isna(df.loc[0, f"normalized_IHA_Group{i}_mean"])


def test_export_scenarios_summary_with_monthly_flows():
    """Test export with monthly released flows"""
    reach = create_test_reach()

    scenario = sc.ConstScenario("Flow Test", "Flow test", reach, [10.0] * 12)
    reach.add_scenario(scenario)
    scenario.compute_QS()

    df = reach.export_scenarios_summary()

    # Check monthly flow averages
    for month in range(1, 13):
        col_name = f"QS_mean_month_{month}_m3s"
        assert col_name in df.columns
        assert not pd.isna(df.loc[0, col_name])


def test_export_scenarios_summary_multiple_scenarios():
    """Test export with multiple scenarios"""
    reach = create_test_reach()

    # Add multiple scenarios
    scenario1 = sc.ConstScenario("Scenario 1", "First", reach, [5.0] * 12)
    scenario2 = sc.ConstScenario("Scenario 2", "Second", reach, [10.0] * 12)
    scenario3 = sc.PropScenario("Scenario 3", "Third", reach, 5.0, 0.2, 3.0, 20.0)

    reach.add_scenario(scenario1)
    reach.add_scenario(scenario2)
    reach.add_scenario(scenario3)

    scenario1.compute_QS()
    scenario2.compute_QS()
    scenario3.compute_QS()

    df = reach.export_scenarios_summary()

    # Check all scenarios are present
    assert len(df) == 3
    assert df.loc[0, "scenario_name"] == "Scenario 1"
    assert df.loc[1, "scenario_name"] == "Scenario 2"
    assert df.loc[2, "scenario_name"] == "Scenario 3"


def test_export_scenarios_summary_csv():
    """Test CSV export"""
    reach = create_test_reach()

    scenario = sc.ConstScenario("CSV Test", "CSV test", reach, [5.0] * 12)
    reach.add_scenario(scenario)
    scenario.compute_QS()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        temp_path = f.name

    try:
        df = reach.export_scenarios_summary(output_path=temp_path, format="csv")

        # Verify file was created
        assert os.path.exists(temp_path)

        # Read it back and verify
        df_read = pd.read_csv(temp_path)
        assert len(df_read) == 1
        assert df_read.loc[0, "scenario_name"] == "CSV Test"
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_export_scenarios_summary_invalid_format():
    """Test that invalid format raises error"""
    reach = create_test_reach()

    scenario = sc.ConstScenario("Format Test", "Format test", reach, [5.0] * 12)
    reach.add_scenario(scenario)
    scenario.compute_QS()

    try:
        reach.export_scenarios_summary(output_path="test.txt", format="txt")
        assert False, "Should raise ValueError for invalid format"
    except ValueError as e:
        assert "csv" in str(e) and "excel" in str(e)


def test_export_scenarios_summary_with_cases_duration():
    """Test export with cases duration"""
    reach = create_test_reach()

    scenario = sc.ConstScenario("Cases Test", "Cases test", reach, [5.0] * 12)
    reach.add_scenario(scenario)
    scenario.compute_QS()

    df = reach.export_scenarios_summary()

    # Check cases duration fields
    assert "case1_duration_fraction" in df.columns
    assert "case2_duration_fraction" in df.columns
    assert "case3_duration_fraction" in df.columns

    # Verify they sum to approximately 1
    total = (
        df.loc[0, "case1_duration_fraction"]
        + df.loc[0, "case2_duration_fraction"]
        + df.loc[0, "case3_duration_fraction"]
    )
    assert abs(total - 1.0) < 0.01


def test_export_scenarios_summary_with_seasonal_volumes():
    """Test export with seasonal volumes"""
    reach = create_test_reach()

    scenario = sc.ConstScenario("Seasonal Test", "Seasonal test", reach, [5.0] * 12)
    reach.add_scenario(scenario)
    scenario.compute_QS()
    scenario.compute_natural_abstracted_volumes()

    df = reach.export_scenarios_summary()

    # Check seasonal volume fields are present
    assert "seasonal_abs_volume_Winter_m3" in df.columns
    assert "seasonal_abs_volume_Spring_m3" in df.columns
    assert "seasonal_abs_volume_Summer_m3" in df.columns
    assert "seasonal_abs_volume_Autumn_m3" in df.columns


def test_export_scenarios_summary_with_annual_sediment_budget():
    """Test export with annual sediment budget"""
    reach = create_test_reach()

    # Add cross-section info for sediment calculations
    reach.add_cross_section_info(
        section=pd.DataFrame({"x [m]": [0, 5, 10], "y [m]": [0, 1, 0]}),
        slope=0.001,
        grain_data=10.0,  # D50 in mm
    )

    scenario = sc.ConstScenario("Sediment Test", "Sediment test", reach, [15.0] * 12)
    reach.add_scenario(scenario)
    scenario.compute_QS()

    # Try to compute annual sediment budget
    scenario.compute_annual_sediment_budget()

    df = reach.export_scenarios_summary()

    # Check sediment budget fields are present
    assert "annual_sediment_budget_total_mean" in df.columns
    assert not pd.isna(df.loc[0, "annual_sediment_budget_total_mean"])
    # There should be a value (could be zero if flow is too small)
    assert df.loc[0, "annual_sediment_budget_total_mean"] >= 0


def test_export_scenarios_summary_excel_missing_openpyxl():
    """Test that Excel export gives helpful error when openpyxl is not installed"""
    reach = create_test_reach()

    scenario = sc.ConstScenario("Excel Test", "Excel test", reach, [5.0] * 12)
    reach.add_scenario(scenario)
    scenario.compute_QS()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".xlsx", delete=False) as f:
        temp_path = f.name

    try:
        # This test assumes openpyxl might not be installed
        # If it is installed, the export will succeed, which is also fine
        try:
            reach.export_scenarios_summary(output_path=temp_path, format="excel")
            # If we get here, openpyxl is installed, so cleanup and pass
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except ImportError as e:
            # This is the expected behavior when openpyxl is not installed
            assert "openpyxl" in str(e)
            assert "pip install" in str(e)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
