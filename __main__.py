import datetime as dt
from index_model.index import IndexModel

if __name__ == "__main__":
    backtest_start = dt.date(2020, 1, 1)
    backtest_end   = dt.date(2020, 12, 31)

    # Replicate old behavior: top-3 with 50/25/25
    index = IndexModel(n_select=3, weights=[0.50, 0.25, 0.25])

    index.calc_index_level(start_date=backtest_start, end_date=backtest_end)
    index.export_values("export.csv")

