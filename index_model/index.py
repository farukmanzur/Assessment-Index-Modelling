import datetime as dt
import pandas as pd


class IndexModel:
    def __init__(self) -> None:
        self.start_level = 100.0
        df = pd.read_csv("data_sources/stock_prices.csv", parse_dates=["Date"], dayfirst=True)
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
        df = df.set_index("Date").sort_index()

        df = df[df.index.weekday < 5].copy() #Ensures there is only weekdays in  the dataframe

        self.prices = df
        self.stocks = [c for c in df.columns if c.startswith("Stock_")] #list of stocks where are going to use later

        self.index_values = None
        self._selection_debug = []

        self._check_missing_weekdays()
        self._check_missing_prices()
        self._check_price_jumps(max_pct_jump=0.2)  # 20% threshold; adjust as needed

       
    def _check_missing_weekdays(self):
        dates = self.prices.index
        expected = pd.bdate_range(start=dates.min(), end=dates.max())
        missing = expected.difference(dates)

        if not missing.empty:
            print("Missing business days detected:")
            print(missing)
            raise ValueError(f"Data has {len(missing)} missing business days between {dates.min().date()} and {dates.max().date()}.")
        print("All business days present — no missing weekdays.")
    
    def _check_missing_prices(self):
        """Verify that there are no missing (NaN) prices in the dataset."""
        # 1. Find NaN values
        nan_mask = self.prices.isna()

        if nan_mask.values.any():
            # 2. Locate them precisely
            missing_positions = nan_mask.stack()[nan_mask.stack()].index  # MultiIndex of (Date, Stock)
            print("Missing price data detected at the following positions:")
            for date, stock in missing_positions:
                print(f"   Date: {date.date()}, Stock: {stock}")
            #if len(missing_positions) > 10:
            
            print(f"Total missing values: {len(missing_positions)}")
            raise ValueError("Data contains missing price values — please fix before continuing.")
        else:
            print("No missing prices found for any stock.")
            
    def _check_price_jumps(self, max_pct_jump: float = 0.20):
        """
        Raise if any absolute daily price change exceeds max_pct_jump (e.g., 0.20 = 20%).
        Checks all stocks and all trading days in self.prices.
        """
        if self.prices is None or self.prices.empty:
            raise ValueError("prices is empty.")

        px = self.prices[self.stocks].copy()
        # daily returns; first row per column is NaN (ignored)
        ret = px.pct_change()

        bad = ret.abs() > max_pct_jump
        if bad.values.any():
            # build a detailed message with every offending (date, stock, return)
            rows, cols = bad.values.nonzero()
            lines = []
            for r, c in zip(rows, cols):
                d = ret.index[r]
                s = ret.columns[c]
                val = ret.iat[r, c]
                if pd.notna(val):
                    lines.append(f"{d.date()} | {s} | {val:.6f} ({val*100:.2f}%)")
            details = "\n".join(lines)
            raise ValueError(
                f"Daily price jump(s) exceed {max_pct_jump*100:.1f}%:\n{details}"
            )
        # if no issues, just return silently
        return True
            

    
    #Groups by year & month, finds earliest date = first business day.
    def _first_business_days(self):
        groups = self.prices.index.to_series().groupby([self.prices.index.year, self.prices.index.month])
        firsts = [g.index.min() for _, g in groups]
        result = sorted(firsts)
        #print("First business days:", result)
        #print(f"First BD: {result}")
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
        result = sel_date
        print(f"First BD: {first_bd.date()} -> Snapshot: {sel_date.date()}")
        return sel_date

    #calculate the index from start date till end date
    def calc_index_level(self, start_date: dt.date, end_date: dt.date) -> None: #Working window
        print(f"Starting index calculation from {start_date} to {end_date}")
        
        start_d = start_date
        end_d = end_date
        trading_dates = [d for d in self.prices.index if start_d <= d.date() <= end_d]
        if not trading_dates:
            raise ValueError("Date error.")

        print(f"Total trading days in window: {len(trading_dates)}")

        first_bds = self._first_business_days() #Sets the dates for rebalance and the date of which stock we are going to pick"
        firstbd_to_snapshot = {}
        for f in first_bds:
            sel = self._selection_snapshot_for_first_bd(f)
            firstbd_to_snapshot[f] = sel

        firstbds_in_window = set([
            f for f in firstbd_to_snapshot.keys() 
            if f in self.prices.index and start_d <= f.date() <= end_d
        ])
        print(f"Number of rebalances in window: {len(firstbds_in_window)}")

        
        level = float(self.start_level)
        shares = {s: 0.0 for s in self.stocks}
        results = []

        pending_shares = None #Bug
        for i, today in enumerate(trading_dates):
            if pending_shares is not None:
                shares = pending_shares
                pending_shares = None
                #print(f"[{today.date()}] Activated new portfolio (from previous FBD).")

            
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

                #print(f"[{today.date()}] Rebalance day:")
                print(f"  Snapshot date: {snapshot_date.date()}")
                print(f"  Top 3 stocks: {top3}")
                #print(f"  Weights: {weights}")

                today_prices = self.prices.loc[today, self.stocks]
                new_shares = {s: 0.0 for s in self.stocks}
                for tkr, w in zip(top3, weights):
                    p = today_prices[tkr]
                    if pd.isna(p) or p == 0:
                        raise ValueError(f"Wrong price for {tkr} on {today.date()}")
                    new_shares[tkr] = (level * w) / p

                pending_shares = new_shares
                next_day = (self.prices.index[self.prices.index > today].min()
                        if any(self.prices.index > today) else None)
                #print(f"  New shares calculated; effective next BD: {next_day.date() if next_day else None}")

                #info for debugging
                self._selection_debug.append({
                    "first_bd": today,
                    "snapshot": snapshot_date,
                    "top3": top3,
                    #"will_apply_on_next_day": True,
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
        #print(df)
        df.to_csv(file_name, index=False)
