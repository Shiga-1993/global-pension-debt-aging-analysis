from __future__ import annotations

import html
import io
import json
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
REPORT_DIR = OUTPUT_DIR / "report"

SOURCE_CHECKED_DATE = date.today().isoformat()

COUNTRIES = {
    "AUS": "Australia",
    "CAN": "Canada",
    "CHE": "Switzerland",
    "DEU": "Germany",
    "DNK": "Denmark",
    "ESP": "Spain",
    "FRA": "France",
    "GBR": "United Kingdom",
    "GRC": "Greece",
    "ITA": "Italy",
    "JPN": "Japan",
    "KOR": "Korea",
    "NLD": "Netherlands",
    "NOR": "Norway",
    "PRT": "Portugal",
    "SWE": "Sweden",
    "USA": "United States",
}

FRED_LONG_TERM_RATE_SERIES = {
    "AUS": "IRLTLT01AUM156N",
    "CAN": "IRLTLT01CAM156N",
    "CHE": "IRLTLT01CHM156N",
    "DEU": "IRLTLT01DEM156N",
    "DNK": "IRLTLT01DKM156N",
    "ESP": "IRLTLT01ESM156N",
    "FRA": "IRLTLT01FRM156N",
    "GBR": "IRLTLT01GBM156N",
    "GRC": "IRLTLT01GRM156N",
    "ITA": "IRLTLT01ITM156N",
    "JPN": "IRLTLT01JPM156N",
    "KOR": "IRLTLT01KRM156N",
    "NLD": "IRLTLT01NLM156N",
    "NOR": "IRLTLT01NOM156N",
    "PRT": "IRLTLT01PTM156N",
    "SWE": "IRLTLT01SEM156N",
    "USA": "IRLTLT01USM156N",
}

PENSION_SPENDING_URL = (
    "https://sdmx.oecd.org/public/rest/v1/data/"
    "OECD.ELS.SPD,DSD_SOCX_AGG@DF_PUB_PRV/"
    ".A..PT_B1GQ.ES20_30%2BES10._T.TP01._Z"
    "?startPeriod=2010&dimensionAtObservation=AllDimensions"
)
WORLD_BANK_OLD_AGE_URL = (
    "https://api.worldbank.org/v2/country/all/indicator/SP.POP.DPND.OL"
    "?format=json&date=2010:2024&per_page=20000"
)
IMF_WEO_DEBT_URL = "https://www.imf.org/external/datamapper/api/v1/GGXWDG_NGDP"


def ensure_dirs() -> None:
    for directory in [FIGURE_DIR, TABLE_DIR, REPORT_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def read_csv_url(url: str, timeout: int = 60) -> pd.DataFrame:
    response = requests.get(url, timeout=timeout, headers={"Accept": "text/csv"})
    response.raise_for_status()
    return pd.read_csv(io.StringIO(response.text))


def fetch_public_pension_spending() -> pd.DataFrame:
    raw = read_csv_url(PENSION_SPENDING_URL)
    data = raw[
        (raw["REF_AREA"].isin(COUNTRIES))
        & (raw["EXPEND_SOURCE"] == "ES10")
        & (raw["PROGRAMME_TYPE"] == "TP01")
        & (raw["UNIT_MEASURE"] == "PT_B1GQ")
    ].copy()
    data["public_pension_spending_pct_gdp"] = pd.to_numeric(data["OBS_VALUE"], errors="coerce")
    data["metric_year"] = pd.to_numeric(data["TIME_PERIOD"], errors="coerce").astype("Int64")
    data = data.dropna(subset=["public_pension_spending_pct_gdp", "metric_year"])
    return data.rename(columns={"REF_AREA": "country_code"})[
        ["country_code", "metric_year", "public_pension_spending_pct_gdp"]
    ].sort_values(["country_code", "metric_year"])


def fetch_old_age_dependency() -> pd.DataFrame:
    response = requests.get(WORLD_BANK_OLD_AGE_URL, timeout=60)
    response.raise_for_status()
    payload = response.json()
    rows = payload[1]
    data = pd.DataFrame(rows)
    data["country_code"] = data["countryiso3code"]
    data["metric_year"] = pd.to_numeric(data["date"], errors="coerce").astype("Int64")
    data["old_age_dependency_ratio"] = pd.to_numeric(data["value"], errors="coerce")
    data = data[data["country_code"].isin(COUNTRIES)].dropna(subset=["metric_year"])
    return data[["country_code", "metric_year", "old_age_dependency_ratio"]].sort_values(
        ["country_code", "metric_year"]
    )


def fetch_imf_government_debt() -> pd.DataFrame:
    response = requests.get(IMF_WEO_DEBT_URL, timeout=60)
    response.raise_for_status()
    values = response.json()["values"]["GGXWDG_NGDP"]
    rows = []
    for country_code, series in values.items():
        if country_code not in COUNTRIES:
            continue
        for year, value in series.items():
            rows.append(
                {
                    "country_code": country_code,
                    "metric_year": int(year),
                    "government_gross_debt_pct_gdp": float(value),
                }
            )
    return pd.DataFrame(rows).sort_values(["country_code", "metric_year"])


def fetch_long_term_bond_yields() -> pd.DataFrame:
    rows = []
    for country_code, series_id in FRED_LONG_TERM_RATE_SERIES.items():
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = pd.read_csv(io.StringIO(response.text))
        data[series_id] = pd.to_numeric(data[series_id].replace(".", pd.NA), errors="coerce")
        data = data.dropna(subset=[series_id]).copy()
        data["observation_date"] = pd.to_datetime(data["observation_date"])
        if data.empty:
            continue
        latest = data.iloc[-1]
        trailing = data.tail(12)
        rows.append(
            {
                "country_code": country_code,
                "bond_yield_latest_date": latest["observation_date"].date().isoformat(),
                "long_term_government_bond_yield_pct": float(latest[series_id]),
                "long_term_government_bond_yield_12m_avg_pct": float(trailing[series_id].mean()),
                "fred_series_id": series_id,
            }
        )
    return pd.DataFrame(rows)


def latest_non_null_by_country(df: pd.DataFrame, value_column: str, year_column: str) -> pd.DataFrame:
    filtered = df.dropna(subset=[value_column]).copy()
    filtered[year_column] = pd.to_numeric(filtered[year_column], errors="coerce")
    filtered = filtered.dropna(subset=[year_column])
    idx = filtered.sort_values(["country_code", year_column]).groupby("country_code").tail(1).index
    return filtered.loc[idx].reset_index(drop=True)


def minmax(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    low = values.min()
    high = values.max()
    if pd.isna(low) or pd.isna(high) or high == low:
        return pd.Series(np.full(len(values), 0.5), index=values.index)
    return (values - low) / (high - low)


def compute_pressure_index(snapshot: pd.DataFrame) -> pd.DataFrame:
    data = snapshot.copy()
    components = {
        "aging_component": "old_age_dependency_ratio",
        "pension_component": "public_pension_spending_pct_gdp",
        "debt_component": "government_gross_debt_pct_gdp",
        "bond_yield_component": "long_term_government_bond_yield_pct",
    }
    for component, source in components.items():
        data[component] = minmax(data[source]) * 25
    data["structural_pressure_index"] = data[list(components)].sum(axis=1)
    data["market_rate_debt_burden_proxy"] = (
        data["government_gross_debt_pct_gdp"] * data["long_term_government_bond_yield_pct"] / 100
    )
    return data.sort_values("structural_pressure_index", ascending=False).reset_index(drop=True)


def build_latest_snapshot(
    pension: pd.DataFrame, aging: pd.DataFrame, debt: pd.DataFrame, yields: pd.DataFrame
) -> pd.DataFrame:
    pension_latest = latest_non_null_by_country(
        pension, "public_pension_spending_pct_gdp", "metric_year"
    ).rename(columns={"metric_year": "pension_spending_year"})
    aging_latest = latest_non_null_by_country(
        aging, "old_age_dependency_ratio", "metric_year"
    ).rename(columns={"metric_year": "old_age_dependency_year"})
    debt_2024 = debt[debt["metric_year"] == 2024].drop(columns=["metric_year"]).copy()
    snapshot = (
        pd.DataFrame({"country_code": list(COUNTRIES)})
        .merge(pension_latest, on="country_code", how="left")
        .merge(aging_latest, on="country_code", how="left")
        .merge(debt_2024, on="country_code", how="left")
        .merge(yields, on="country_code", how="left")
    )
    snapshot["country"] = snapshot["country_code"].map(COUNTRIES)
    ordered = [
        "country_code",
        "country",
        "old_age_dependency_ratio",
        "old_age_dependency_year",
        "public_pension_spending_pct_gdp",
        "pension_spending_year",
        "government_gross_debt_pct_gdp",
        "long_term_government_bond_yield_pct",
        "long_term_government_bond_yield_12m_avg_pct",
        "bond_yield_latest_date",
        "fred_series_id",
    ]
    snapshot = snapshot[ordered]
    required = [
        "old_age_dependency_ratio",
        "public_pension_spending_pct_gdp",
        "government_gross_debt_pct_gdp",
        "long_term_government_bond_yield_pct",
    ]
    return snapshot.dropna(subset=required).reset_index(drop=True)


def build_aging_pension_trajectory(pension: pd.DataFrame, aging: pd.DataFrame) -> pd.DataFrame:
    trajectory = pension.merge(aging, on=["country_code", "metric_year"], how="inner")
    trajectory["country"] = trajectory["country_code"].map(COUNTRIES)
    return trajectory.sort_values(["country_code", "metric_year"]).reset_index(drop=True)


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (10.8, 6.4),
            "figure.dpi": 150,
            "savefig.dpi": 180,
            "font.size": 10.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.26,
            "axes.axisbelow": True,
            "legend.frameon": False,
        }
    )


def label_points(
    ax: plt.Axes,
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_offset: float,
    y_offset: float,
    offsets: dict[str, tuple[float, float]] | None = None,
) -> None:
    offsets = offsets or {}
    for _, row in data.iterrows():
        dx, dy = offsets.get(row["country_code"], (x_offset, y_offset))
        ax.text(
            row[x_col] + dx,
            row[y_col] + dy,
            row["country_code"],
            fontsize=8.5,
            color="#263238",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.35},
            zorder=5,
        )


def save_figure(fig: plt.Figure, filename: str) -> Path:
    path = FIGURE_DIR / filename
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_aging_pension_bubble(snapshot: pd.DataFrame) -> Path:
    fig, ax = plt.subplots()
    sizes = 90 + snapshot["government_gross_debt_pct_gdp"] * 3.8
    scatter = ax.scatter(
        snapshot["old_age_dependency_ratio"],
        snapshot["public_pension_spending_pct_gdp"],
        s=sizes,
        c=snapshot["long_term_government_bond_yield_pct"],
        cmap="viridis",
        alpha=0.82,
        edgecolor="white",
        linewidth=0.8,
    )
    label_points(
        ax,
        snapshot,
        "old_age_dependency_ratio",
        "public_pension_spending_pct_gdp",
        x_offset=0.35,
        y_offset=0.05,
        offsets={
            "GRC": (0.35, 0.16),
            "ITA": (0.35, -0.18),
            "GBR": (0.35, -0.05),
            "NLD": (0.35, -0.12),
            "CHE": (0.35, -0.08),
        },
    )
    ax.set_xlabel("Old-age dependency ratio, 65+ per 100 working-age people")
    ax.set_ylabel("Public pension spending (% of gross domestic product)")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Latest 10-year government bond yield (%)")
    return save_figure(fig, "aging_pension_debt_yield_bubble.png")


def plot_debt_yield_quadrants(snapshot: pd.DataFrame) -> Path:
    fig, ax = plt.subplots()
    sizes = 80 + snapshot["public_pension_spending_pct_gdp"] * 35
    scatter = ax.scatter(
        snapshot["government_gross_debt_pct_gdp"],
        snapshot["long_term_government_bond_yield_pct"],
        s=sizes,
        c=snapshot["old_age_dependency_ratio"],
        cmap="plasma",
        alpha=0.82,
        edgecolor="white",
        linewidth=0.8,
    )
    label_points(
        ax,
        snapshot,
        "government_gross_debt_pct_gdp",
        "long_term_government_bond_yield_pct",
        x_offset=2.0,
        y_offset=0.03,
        offsets={
            "DNK": (2.0, 0.06),
            "SWE": (2.0, -0.07),
            "FRA": (2.0, 0.11),
            "CAN": (2.0, 0.11),
            "ESP": (2.0, -0.12),
            "PRT": (-8.0, -0.02),
            "GRC": (2.0, 0.05),
            "ITA": (2.0, 0.03),
        },
    )
    ax.axvline(snapshot["government_gross_debt_pct_gdp"].median(), color="#6b7280", lw=1.1, ls="--")
    ax.axhline(snapshot["long_term_government_bond_yield_pct"].median(), color="#6b7280", lw=1.1, ls="--")
    ax.set_xlabel("General government gross debt (% of gross domestic product), 2024")
    ax.set_ylabel("Latest 10-year government bond yield (%)")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Old-age dependency ratio")
    return save_figure(fig, "debt_yield_pension_quadrants.png")


def plot_aging_pension_trajectory(trajectory: pd.DataFrame) -> Path:
    selected = ["JPN", "ITA", "FRA", "DEU", "KOR", "USA", "NLD", "AUS"]
    palette = {
        "JPN": "#b42318",
        "ITA": "#ca8a04",
        "FRA": "#2563eb",
        "DEU": "#111827",
        "KOR": "#7c3aed",
        "USA": "#1f7a8c",
        "NLD": "#0f766e",
        "AUS": "#8c564b",
    }
    fig, ax = plt.subplots()
    for country_code in selected:
        sub = trajectory[trajectory["country_code"] == country_code].sort_values("metric_year")
        if sub.empty:
            continue
        ax.plot(
            sub["old_age_dependency_ratio"],
            sub["public_pension_spending_pct_gdp"],
            marker="o",
            lw=2,
            ms=4,
            label=COUNTRIES[country_code],
            color=palette[country_code],
        )
        last = sub.iloc[-1]
        ax.text(
            last["old_age_dependency_ratio"] + 0.25,
            last["public_pension_spending_pct_gdp"],
            country_code,
            fontsize=8.5,
            color=palette[country_code],
        )
    ax.set_xlabel("Old-age dependency ratio, 65+ per 100 working-age people")
    ax.set_ylabel("Public pension spending (% of gross domestic product)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4)
    return save_figure(fig, "aging_pension_trajectory.png")


def plot_structural_pressure_rank(scored: pd.DataFrame) -> Path:
    ordered = scored.sort_values("structural_pressure_index", ascending=True).copy()
    y = np.arange(len(ordered))
    fig, ax = plt.subplots(figsize=(11.2, 7.2))
    colors = {
        "aging_component": "#1f7a8c",
        "pension_component": "#b42318",
        "debt_component": "#ca8a04",
        "bond_yield_component": "#7c3aed",
    }
    left = np.zeros(len(ordered))
    labels = {
        "aging_component": "Aging",
        "pension_component": "Public pension spending",
        "debt_component": "Government debt",
        "bond_yield_component": "Bond yield",
    }
    for component in labels:
        values = ordered[component].to_numpy()
        ax.barh(y, values, left=left, color=colors[component], label=labels[component])
        left += values
    ax.set_yticks(y, ordered["country"])
    ax.set_xlabel("Structural pressure index, equal-weight normalized components")
    ax.set_xlim(0, 105)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=4)
    for i, value in enumerate(ordered["structural_pressure_index"]):
        ax.text(value + 1.2, i, f"{value:.0f}", va="center", fontsize=8.5)
    return save_figure(fig, "structural_pressure_rank.png")


def format_pct(value: float, digits: int = 1) -> str:
    return "" if pd.isna(value) else f"{value:.{digits}f}%"


def format_num(value: float, digits: int = 1) -> str:
    return "" if pd.isna(value) else f"{value:.{digits}f}"


def html_table(df: pd.DataFrame, columns: list[tuple[str, str]], limit: int | None = None) -> str:
    rows = df.head(limit) if limit else df
    header = "".join(f"<th>{html.escape(label)}</th>" for _, label in columns)
    body_rows = []
    for _, row in rows.iterrows():
        cells = []
        for key, _ in columns:
            value = row[key]
            cells.append(f"<td>{html.escape(str(value))}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def build_report(scored: pd.DataFrame, trajectory: pd.DataFrame, quality: pd.DataFrame) -> Path:
    top = scored.iloc[0]
    japan = scored[scored["country_code"] == "JPN"].iloc[0]
    korea = scored[scored["country_code"] == "KOR"].iloc[0]
    italy = scored[scored["country_code"] == "ITA"].iloc[0]
    latest_display = scored.copy()
    latest_display["old_age_dependency_ratio"] = latest_display["old_age_dependency_ratio"].map(format_num)
    latest_display["public_pension_spending_pct_gdp"] = latest_display[
        "public_pension_spending_pct_gdp"
    ].map(format_pct)
    latest_display["government_gross_debt_pct_gdp"] = latest_display[
        "government_gross_debt_pct_gdp"
    ].map(format_pct)
    latest_display["long_term_government_bond_yield_pct"] = latest_display[
        "long_term_government_bond_yield_pct"
    ].map(format_pct)
    latest_display["structural_pressure_index"] = latest_display["structural_pressure_index"].map(
        lambda x: f"{x:.0f}"
    )
    latest_display["pension_spending_year"] = latest_display["pension_spending_year"].astype(str)

    quality_display = quality.copy()
    quality_display["value"] = quality_display["value"].astype(str)

    report_html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Global Pension, Aging, Government Debt, and Bond Yield Analysis</title>
  <style>
    :root {{
      --ink: #18212f;
      --muted: #586576;
      --line: #dce3ea;
      --panel: #f7f9fb;
      --accent: #1f7a8c;
    }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: white;
      line-height: 1.58;
    }}
    main {{
      max-width: 1120px;
      margin: 0 auto;
      padding: 42px 22px 72px;
    }}
    h1 {{
      font-size: clamp(30px, 4vw, 48px);
      line-height: 1.08;
      margin: 0 0 12px;
      letter-spacing: 0;
    }}
    h2 {{
      font-size: 24px;
      margin: 38px 0 12px;
      padding-top: 24px;
      border-top: 1px solid var(--line);
    }}
    h3 {{
      font-size: 18px;
      margin: 0 0 12px;
    }}
    p, li {{
      font-size: 16px;
    }}
    .lede {{
      max-width: 880px;
      font-size: 18px;
      color: var(--muted);
    }}
    .summary-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
      gap: 12px;
      margin: 26px 0;
    }}
    .metric {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px 16px;
    }}
    .metric-label {{
      color: var(--muted);
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: .04em;
    }}
    .metric-value {{
      font-size: 24px;
      font-weight: 700;
      margin-top: 4px;
    }}
    figure {{
      margin: 28px 0 34px;
    }}
    figure img {{
      max-width: 100%;
      height: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: white;
    }}
    figcaption {{
      margin-top: 8px;
      color: var(--muted);
      font-size: 14px;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      font-size: 14px;
      margin: 14px 0 22px;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      text-align: left;
      padding: 8px 10px;
      vertical-align: top;
    }}
    th {{
      background: var(--panel);
      font-weight: 650;
    }}
    .note {{
      color: var(--muted);
      font-size: 14px;
    }}
    a {{
      color: var(--accent);
    }}
  </style>
</head>
<body>
<main>
  <h1>Global Pension, Aging, Government Debt, and Bond Yield Analysis</h1>
  <p class="lede">A country-level structural comparison of old-age dependency, public pension spending, general government gross debt, and long-term government bond yields.</p>

  <div class="summary-grid">
    <div class="metric"><div class="metric-label">Countries compared</div><div class="metric-value">{len(scored)}</div></div>
    <div class="metric"><div class="metric-label">Highest index</div><div class="metric-value">{html.escape(top['country'])}</div></div>
    <div class="metric"><div class="metric-label">Japan profile</div><div class="metric-value">{japan['structural_pressure_index']:.0f}/100</div></div>
    <div class="metric"><div class="metric-label">Source checked</div><div class="metric-value">{SOURCE_CHECKED_DATE}</div></div>
  </div>

  <h2>Background</h2>
  <p>Public pension systems sit at the intersection of demographics and government finance. A country can have high old-age dependency but low bond yields, or high public pension spending but moderate government debt. Looking at one indicator alone can hide the real comparison.</p>

  <h2>Objective</h2>
  <p>The objective is to compare countries across four linked dimensions: old-age dependency, public pension spending, general government gross debt, and long-term government bond yields. The output is a structural dashboard, not a pension forecast or investment recommendation.</p>

  <h2>Method</h2>
  <ul>
    <li>Public pension spending uses Organisation for Economic Co-operation and Development public and private social expenditure data for old-age programmes, public expenditure source, measured as a percentage of gross domestic product.</li>
    <li>Old-age dependency uses the World Bank indicator for people aged 65 and older relative to the working-age population.</li>
    <li>Government debt uses International Monetary Fund World Economic Outlook DataMapper general government gross debt as a percentage of gross domestic product for 2024.</li>
    <li>Bond yields use Federal Reserve Bank of St. Louis series sourced from Organisation for Economic Co-operation and Development long-term government bond yields.</li>
    <li>The structural pressure index normalizes the four indicators from low to high within this country set and gives each component equal weight.</li>
  </ul>

  <h2>Key Findings</h2>
  <ul>
    <li>{html.escape(top['country'])} ranks highest in the simple structural pressure index because multiple dimensions are high at the same time.</li>
    <li>Japan combines the highest old-age dependency and very high gross government debt, but its latest 10-year government bond yield is still below many peers.</li>
    <li>Italy combines high public pension spending, high debt, and a higher bond yield than Japan, which makes its position look structurally tighter on the market-rate view.</li>
    <li>Korea currently has much lower public pension spending than older European systems, but its old-age dependency has been rising quickly.</li>
  </ul>

  <h2>Figures</h2>
  <figure>
    <img src="../figures/aging_pension_debt_yield_bubble.png" alt="Bubble chart comparing old-age dependency, public pension spending, government debt, and bond yields">
    <figcaption>Old-age dependency and public pension spending show the demographic and fiscal side; bubble size and color add debt and market yield context.</figcaption>
  </figure>
  <figure>
    <img src="../figures/debt_yield_pension_quadrants.png" alt="Scatter plot comparing government debt and long-term government bond yields">
    <figcaption>Debt-to-output ratios look different once the long-term government bond yield is added.</figcaption>
  </figure>
  <figure>
    <img src="../figures/aging_pension_trajectory.png" alt="Country trajectories for old-age dependency and public pension spending">
    <figcaption>Selected country trajectories show whether systems are moving along the same path or diverging.</figcaption>
  </figure>
  <figure>
    <img src="../figures/structural_pressure_rank.png" alt="Structural pressure index ranking with components">
    <figcaption>The index is an equal-weight comparison of four normalized indicators, not a risk model.</figcaption>
  </figure>

  <h2>Country Snapshot</h2>
  {html_table(
        latest_display,
        [
            ("country", "Country"),
            ("old_age_dependency_ratio", "Old-age dependency"),
            ("public_pension_spending_pct_gdp", "Public pension spending"),
            ("pension_spending_year", "Pension spending year"),
            ("government_gross_debt_pct_gdp", "Government gross debt"),
            ("long_term_government_bond_yield_pct", "Latest 10-year bond yield"),
            ("structural_pressure_index", "Index"),
        ],
    )}

  <h2>Data Quality Checks</h2>
  {html_table(quality_display, [("check", "Check"), ("value", "Value"), ("note", "Note")])}

  <h2>Discussion</h2>
  <p>The main result is that high debt is not the same thing as high market pressure, and high aging is not the same thing as high current pension spending. Japan and Italy are useful contrasts: both are old and heavily indebted, but their bond yield environments are different. Korea is a different case: the current public pension spending line is lower, but the demographic path is moving quickly.</p>

  <h2>Limitations</h2>
  <ul>
    <li>The index is descriptive and depends on the selected countries and selected indicators.</li>
    <li>Pension spending years are not identical across countries, so the latest-country snapshot mixes latest available pension years.</li>
    <li>Long-term government bond yields are latest market observations, not average interest cost on the debt stock.</li>
    <li>Gross government debt ignores public assets and other balance-sheet differences.</li>
    <li>The analysis does not model tax capacity, productivity growth, migration, retirement age, benefit formulas, or private household wealth.</li>
  </ul>

  <h2>Sources</h2>
  <ul>
    <li><a href="https://www.oecd.org/en/data/indicators/pension-spending.html">Organisation for Economic Co-operation and Development pension spending indicator</a></li>
    <li><a href="https://api.worldbank.org/v2/country/all/indicator/SP.POP.DPND.OL?format=json">World Bank old-age dependency ratio data endpoint</a></li>
    <li><a href="https://www.imf.org/external/datamapper/datasets/WEO">International Monetary Fund World Economic Outlook DataMapper</a></li>
    <li><a href="https://www.oecd.org/en/data/indicators/long-term-interest-rates.html">Organisation for Economic Co-operation and Development long-term interest rates indicator</a></li>
    <li><a href="https://fred.stlouisfed.org/">Federal Reserve Bank of St. Louis economic data service</a></li>
  </ul>
</main>
</body>
</html>
"""
    path = REPORT_DIR / "review_report.html"
    path.write_text(report_html, encoding="utf-8")
    return path


def build_quality_summary(
    pension: pd.DataFrame, aging: pd.DataFrame, debt: pd.DataFrame, yields: pd.DataFrame, snapshot: pd.DataFrame
) -> pd.DataFrame:
    rows = [
        {
            "check": "countries_configured",
            "value": len(COUNTRIES),
            "note": "Countries requested in the first-pass comparison set.",
        },
        {
            "check": "countries_with_complete_snapshot",
            "value": len(snapshot),
            "note": "Countries with non-missing aging, pension spending, debt, and yield fields.",
        },
        {
            "check": "public_pension_spending_rows",
            "value": len(pension),
            "note": "Organisation for Economic Co-operation and Development public old-age pension spending observations.",
        },
        {
            "check": "old_age_dependency_rows",
            "value": len(aging),
            "note": "World Bank old-age dependency observations for selected countries.",
        },
        {
            "check": "government_debt_rows",
            "value": len(debt),
            "note": "International Monetary Fund general government gross debt observations for selected countries.",
        },
        {
            "check": "bond_yield_countries",
            "value": len(yields),
            "note": "Countries with a fetched long-term government bond yield series.",
        },
        {
            "check": "source_checked_date",
            "value": SOURCE_CHECKED_DATE,
            "note": "Date the local script retrieved public source data.",
        },
    ]
    return pd.DataFrame(rows)


def main() -> None:
    ensure_dirs()
    configure_plot_style()

    pension = fetch_public_pension_spending()
    aging = fetch_old_age_dependency()
    debt = fetch_imf_government_debt()
    yields = fetch_long_term_bond_yields()
    snapshot = build_latest_snapshot(pension, aging, debt, yields)
    scored = compute_pressure_index(snapshot)
    trajectory = build_aging_pension_trajectory(pension, aging)
    quality = build_quality_summary(pension, aging, debt, yields, scored)

    scored.to_csv(TABLE_DIR / "latest_country_snapshot.csv", index=False)
    trajectory.to_csv(TABLE_DIR / "aging_pension_trajectory.csv", index=False)
    quality.to_csv(TABLE_DIR / "quality_summary.csv", index=False)
    (TABLE_DIR / "run_metadata.json").write_text(
        json.dumps(
            {
                "source_checked_date": SOURCE_CHECKED_DATE,
                "countries": COUNTRIES,
                "pension_spending_url": PENSION_SPENDING_URL,
                "world_bank_old_age_url": WORLD_BANK_OLD_AGE_URL,
                "imf_weo_debt_url": IMF_WEO_DEBT_URL,
                "fred_series": FRED_LONG_TERM_RATE_SERIES,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    plot_aging_pension_bubble(scored)
    plot_debt_yield_quadrants(scored)
    plot_aging_pension_trajectory(trajectory)
    plot_structural_pressure_rank(scored)
    report_path = build_report(scored, trajectory, quality)
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
