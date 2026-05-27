from run_pension_debt_aging_analysis import compute_pressure_index, latest_non_null_by_country


def test_pressure_index_is_bounded(sample_snapshot):
    scored = compute_pressure_index(sample_snapshot)
    assert scored["structural_pressure_index"].between(0, 100).all()


def test_latest_non_null_by_country_keeps_latest_year(sample_time_series):
    latest = latest_non_null_by_country(sample_time_series, "value", "metric_year")
    assert latest.loc[latest["country_code"] == "AAA", "metric_year"].iloc[0] == 2021
    assert latest.loc[latest["country_code"] == "BBB", "metric_year"].iloc[0] == 2020


def test_pressure_index_ranks_higher_constraints_above_lower(sample_snapshot):
    scored = compute_pressure_index(sample_snapshot)
    high = scored.loc[scored["country_code"] == "HIGH", "structural_pressure_index"].iloc[0]
    low = scored.loc[scored["country_code"] == "LOW", "structural_pressure_index"].iloc[0]
    assert high > low


def pytest_generate_tests(metafunc):
    import pandas as pd

    if "sample_snapshot" in metafunc.fixturenames:
        metafunc.parametrize(
            "sample_snapshot",
            [
                pd.DataFrame(
                    [
                        {
                            "country_code": "HIGH",
                            "old_age_dependency_ratio": 50,
                            "public_pension_spending_pct_gdp": 14,
                            "government_gross_debt_pct_gdp": 160,
                            "long_term_government_bond_yield_pct": 5,
                        },
                        {
                            "country_code": "LOW",
                            "old_age_dependency_ratio": 20,
                            "public_pension_spending_pct_gdp": 4,
                            "government_gross_debt_pct_gdp": 30,
                            "long_term_government_bond_yield_pct": 1,
                        },
                    ]
                )
            ],
        )
    if "sample_time_series" in metafunc.fixturenames:
        metafunc.parametrize(
            "sample_time_series",
            [
                pd.DataFrame(
                    [
                        {"country_code": "AAA", "metric_year": 2020, "value": 1.0},
                        {"country_code": "AAA", "metric_year": 2021, "value": 2.0},
                        {"country_code": "BBB", "metric_year": 2020, "value": 3.0},
                        {"country_code": "BBB", "metric_year": 2021, "value": None},
                    ]
                )
            ],
        )
