# Global Pension, Aging, Government Debt, and Bond Yield Analysis

This repository contains a reproducible country-level comparison of aging, public pension spending, government debt, and long-term government bond yields.

## Report

- [Open the rendered web report](https://shiga-1993.github.io/global-pension-debt-aging-analysis/)

Opening `index.html` directly on GitHub shows the source. Use the GitHub Pages link above to view the rendered report.

## Analytical Question

Which countries look structurally tight when old-age dependency, public pension spending, government debt, and long-term government bond yields are viewed together?

The report is a descriptive structural comparison. It is not investment advice, pension advice, legal advice, or a solvency forecast.

## Main Takeaways

- Japan stands out because old-age dependency and gross government debt are both very high.
- Greece and Italy combine high public pension spending, high government debt, and higher long-term government bond yields than Japan.
- Korea has lower current public pension spending than older European systems, but its old-age dependency has been rising quickly.
- High government debt is not the same thing as high market pressure, and high aging is not the same thing as high current pension spending.

## Data Sources

- Organisation for Economic Co-operation and Development public and private social expenditure data: public old-age pension spending as a percentage of gross domestic product.
- World Bank: old-age dependency ratio.
- International Monetary Fund World Economic Outlook DataMapper: general government gross debt as a percentage of gross domestic product.
- Federal Reserve Bank of St. Louis series sourced from Organisation for Economic Co-operation and Development long-term government bond yield data.

## Method

The pipeline downloads selected public series, keeps a focused set of advanced economies and pension-system reference countries, builds a latest-country snapshot, creates trajectory and ranking figures, and renders a standalone web report.

Heavy raw source files are not committed to this repository.

## Repository Contents

- `index.html`: rendered GitHub Pages report
- `run_pension_debt_aging_analysis.py`: reproducible analysis script
- `outputs/figures/`: generated report figures
- `outputs/tables/`: aggregate summary tables and run metadata
- `data/README.md`: data handling notes
- `tests/`: focused metric tests

## Reproduce

From this repository:

```bash
python3 -m pip install --user -r requirements.txt
python3 run_pension_debt_aging_analysis.py
python3 -m pytest -q
```

The script writes report outputs under `outputs/` without storing heavy raw data files.

## Limitations

- The structural pressure index is descriptive and depends on the selected countries and selected indicators.
- Pension spending years differ by country because latest available source coverage is uneven.
- Latest bond yields are market observations from 2026, while debt and demographic indicators are mostly 2024 and pension spending is generally earlier.
- Government debt is gross debt, not net debt.
- Long-term yields do not measure actual average interest cost on outstanding government debt.
