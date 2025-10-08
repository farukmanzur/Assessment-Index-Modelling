import datetime as dt
import pandas as pd


class IndexModel:
    def __init__(self) -> None:
        self.start_level = 100.0
        df = pd.read_csv("data_sources/stock_prices.csv", parse_dates=["Date"], dayfirst=True)
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
        df = df.set_index("Date").sort_index()

        df = df[df.index.weekday < 5].copy()

        self.prices = df
        self.stocks = [c for c in df.columns if c.startswith("Stock_")]

        self.index_values = None
        self._selection_debug = []

    #Groups by year & month, finds earliest date = first business day.
    def _first_business_days(self):
        groups = self.prices.index.to_series().groupby([self.prices.index.year, self.prices.index.month])
        firsts = [g.index.min() for _, g in groups]
        return sorted(firsts)

    #Gets snapshot used for stock selection and return the last buesiness day of the previous month, otherwise point to the earliest
    def _selection_snapshot_for_first_bd(self, first_bd):
        prev_month_day = (first_bd.replace(day=1) - pd.Timedelta(days=1))
        mask = (self.prices.index.year == prev_month_day.year) & (self.prices.index.month == prev_month_day.month)
        if mask.any():
            sel_date = self.prices.index[mask].max()
        else:
            earlier = self.prices.index[self.prices.index <= first_bd]
            if earlier.empty:
                sel_date = self.prices.index.min()
            else:
                sel_date = earlier.max()
        return sel_date

    #calculate the index from start date till end date
    def calc_index_level(self, start_date: dt.date, end_date: dt.date) -> None:
        start_d = start_date
        end_d = end_date
        trading_dates = [d for d in self.prices.index if start_d <= d.date() <= end_d]
        if not trading_dates:
            raise ValueError("Date error.")

        first_bds = self._first_business_days()
        firstbd_to_snapshot = {}
        for f in first_bds:
            sel = self._selection_snapshot_for_first_bd(f)
            firstbd_to_snapshot[f] = sel

        firstbds_in_window = set([f for f in firstbd_to_snapshot.keys() if f in self.prices.index and start_d <= f.date() <= end_d])

        
        level = float(self.start_level)
        shares = {s: 0.0 for s in self.stocks}
        results = []
        
        for i, today in enumerate(trading_dates):
            if sum(shares.values()) > 0:
                # portfolio valuation
                today_prices = self.prices.loc[today, self.stocks]
                level = sum(shares[s] * today_prices[s] for s in self.stocks)
            else:
                level = level

            results.append({"Date": today.strftime("%d/%m/%Y"), "Index_Level": float(level)})

            if today in firstbds_in_window:
                snapshot_date = firstbd_to_snapshot[today]
                snap = self.prices.loc[snapshot_date, self.stocks]

                tmp = pd.DataFrame({"ticker": snap.index, "price": snap.values})
                tmp = tmp.sort_values(by=["price", "ticker"], ascending=[False, True])
                top3 = tmp["ticker"].iloc[:3].tolist()
                weights = [0.5, 0.25, 0.25]

                today_prices = self.prices.loc[today, self.stocks]
                new_shares = {s: 0.0 for s in self.stocks}
                for tkr, w in zip(top3, weights):
                    p = today_prices[tkr]
                    if pd.isna(p) or p == 0:
                        raise ValueError(f"Wrong price for {tkr} on {today.date()}")
                    new_shares[tkr] = (level * w) / p

                #info for debugging
                self._selection_debug.append({
                    "first_bd": today,
                    "snapshot": snapshot_date,
                    "top3": top3,
                    "effective_from": (self.prices.index[self.prices.index > today].min()
                                       if any(self.prices.index > today) else None)
                })

        #index level, with all decimals
        self.index_values = pd.DataFrame(results)

    def export_values(self, file_name: str) -> None:
        if self.index_values is None:
            raise RuntimeError("Call calc_index_level(...) before export_values(...)")
        df = self.index_values.copy()
        df["Index_Level"] = df["Index_Level"].round(2)
        df.to_csv(file_name, index=False)