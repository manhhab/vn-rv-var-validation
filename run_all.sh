#!/usr/bin/env bash
# Reproduces all results in the paper. Run from the repository root.
# Uses data/VNINDEX_1min.csv.gz and data/VN30_1min.csv.gz (included).
set -e
python3 ana/load.py                    # read xlsx, cache as pickle
python3 ana/audit.py                   # data audit: timestamps, sessions
python3 ana/audit3.py                  # record format before/after 2025-05-05
python3 ana/buildB.py                  # regular 1-minute grid, continuous sessions
python3 ana/dailyB.py                  # daily realized measures and benchmark
python3 ana/ciB.py                     # bias by year / regime, bootstrap CIs
python3 ana/tables.py                  # Table 1 data (signature plot, bias, CIs)
python3 ana/quint.py                   # ex ante vs same-day volatility quintiles
for W in 400 250 300 500; do
  python3 ana/runB.py forecast $W      # HAR / GARCH forecasts and all backtests
done
python3 ana/compare.py Bforecast400    # pooled rejections, MCS, DM, exception concentration
python3 ana/collect.py                 # writes data/results.json (Tables 2-5)
python3 ana/tmap.py VNINDEX            # Student t mappings (Table 7)
python3 ana/tmap.py VN30
python3 ana/intraday.py                # intraday autocorrelation, recording-format regimes
python3 ana/simtab4.py                 # simulation of the mechanism (Section 4.2)
python3 ana/mc.py                      # Monte Carlo size and power (Section 4.5)
python3 ana/mc2.py
