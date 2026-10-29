# Realized volatility, nonsynchronous trading and VaR model validation in a frontier market

Replication code for

> Tran, M. H., and M. N. Tran. "Realized volatility, nonsynchronous trading and value-at-risk model validation in a frontier market." Manuscript submitted for publication.

The code measures the bias of realized variance for the VNINDEX and VN30 indexes of the Ho Chi Minh Stock Exchange
(one-minute data, June 2022 – September 2026), and evaluates HAR and GARCH value-at-risk / expected-shortfall forecasts
under a Gaussian mapping, filtered historical simulation and Student t mappings, with a full battery of backtests,
the FZ0 consistent loss, Diebold–Mariano tests and the model confidence set.

## Data

The one-minute index records for the VNINDEX and VN30 (provided by FiinGroup) are included in `data/` as compressed CSV
files and are free to use for research and replication. See `data/README.md` for the variables and notes on the raw records.
If you use the data, please cite the paper and acknowledge FiinGroup as the data source.

## Requirements

Python 3.10 or later and the packages in `requirements.txt`:

    pip install -r requirements.txt

## Reproducing the results

From the repository root:

    ./run_all.sh

All intermediate and output files are written to `data/`. The full run, including the Monte Carlo simulations, takes
roughly one to two hours on a laptop.

| Script | Purpose | Paper |
|---|---|---|
| `ana/load.py` | Read the one-minute data files | – |
| `ana/audit.py`, `ana/audit2.py`, `ana/audit3.py` | Data audit: timestamps, session structure, change in recording format on 5 May 2025 | Section 3.1 |
| `ana/buildB.py` | Regular one-minute grid for the continuous sessions 9:15–11:30 and 13:00–14:30 (previous tick) | Section 3.1 |
| `ana/measures.py` | Realized variance, RV-AC(q), realized kernel (Parzen, BNHLS bandwidth), two-scale estimator, bipower variation | Section 3.3 |
| `ana/dailyB.py` | Daily realized measures and the session benchmark b(t) | Section 3.3 |
| `ana/tables.py`, `ana/ciB.py` | Signature plot, bias, stationary block-bootstrap confidence intervals, bias by year and regime | Table 1, Sections 4.1, 4.7 |
| `ana/quint.py` | Autocorrelation and shortfall by ex ante and same-day volatility quintiles | Section 4.2 |
| `ana/simtab4.py` | Simulation of index staleness, independent noise and nonsynchronous trading | Section 4.2 |
| `ana/forecast.py` | HAR and GARCH(1,1) skewed-t forecasts; Gaussian and filtered historical simulation mappings | Section 3.4 |
| `ana/tests.py` | Kupiec, Christoffersen, DQ, Acerbi–Székely, Du–Escanciano, multinomial tests, Basel zones, FZ0 loss, Diebold–Mariano, model confidence set | Section 3.6 |
| `ana/evalrun.py`, `ana/runB.py` | Rolling forecasts and backtests for estimation windows of 250–500 days | Tables 2–3, Section 4.7 |
| `ana/compare.py`, `ana/collect.py` | Pooled rejection rates, MCS, Diebold–Mariano statistics, exception concentration | Tables 2–5 |
| `ana/tmap.py` | Student t mappings (unit variance and estimated scale) | Table 7, Section 4.6 |
| `ana/mc.py`, `ana/mc2.py` | Monte Carlo size and power of the backtests | Section 4.5 |
| `ana/intraday.py` | Intraday autocorrelation and comparison across recording formats | Section 4.7 |

## License

Code: MIT (see `LICENSE`). Data: provided by FiinGroup and shared free of charge for research and replication; please acknowledge FiinGroup as the source.
