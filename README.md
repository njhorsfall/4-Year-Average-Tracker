# 4-Year Average Tracker

Tracks every **S&P 500** and **FTSE 100** stock against its **4-year average price**, which this strategy treats as the stock's fair price.

| Signal | Rule |
|---|---|
| ▼ **BUY** | latest close ≤ 4-year average **and** earnings healthy |
| ◆ **WATCH** | latest close ≤ 4-year average, but earnings weak or no data (possible value trap) |
| **HOLD** | between the average and the sell line |
| ▲ **SELL** | latest close ≥ 4-year average × (1 + threshold) |

The sell threshold (**+10% / +20% / +30%**) is a toggle on the page, so you can compare all three without re-running anything. Stocks with less than ~3.6 years of history show **N/A**.

**Healthy earnings** means passing every check that has data:

1. **Profitable:** latest annual earnings per share (EPS) is above zero.
2. **Growing:** latest annual EPS is at least as high as three years earlier.
3. **No recent drop:** latest annual EPS is at least 90% of the previous year's.
4. **Analysts expect growth:** the next-year EPS estimate is at or above the last 12 months. This is skipped if there's no estimate.

Checks 1 and 2 must have data. Each stock also shows an approximate *P/E vs its 4-year norm*, so you can tell whether it's cheap relative to its profits as well as its price. Earnings are only downloaded for stocks at, below or within 5% of their average, and are cached for a week (`fundamentals_cache.json`) to keep runs quick.

Tap any stock to see its price, the buy and sell prices, how far it has to move to reach each one, and a 4-year chart with the average and sell lines.

## Option A: run it on your computer

```bash
pip install -r requirements.txt
python tracker.py          # about 5–10 minutes on first run, quicker after
open docs/index.html       # or double-click it
```

`python tracker.py --demo` builds the page from made-up prices, with no internet needed.

## Option B: free hosted app that updates itself (recommended)

1. Create a new **GitHub** repository and upload every file in this folder, including `.github/workflows/update.yml`.
2. In the repo, open **Settings → Pages**. Set *Source* to **Deploy from a branch**, then choose **main** and **/docs**.
3. Open **Actions → Update dashboard → Run workflow** to build it the first time.
4. Your dashboard will be at `https://<your-username>.github.io/<repo-name>/`. Add it to your phone's home screen and it works like an app.

After that it refreshes automatically at 22:17 UK time every weekday, after both markets have closed.

> Note: GitHub Pages sites on free accounts are public. The page only contains public market data.

## Customising

- **Extra stocks:** copy `watchlist.example.csv` to `watchlist.csv` and add Yahoo tickers. London stocks end in `.L`.
- **Averaging window:** change `YEARS = 4` in `tracker.py`.
- **History needed before a signal is given:** `MIN_COVERAGE`.
- **Earnings rules:** `earnings_health()` in `tracker.py`, plus the `RECENT_DROP_LIMIT` and `FUND_SCOPE` settings.

## Details and caveats

- Data comes from Yahoo Finance via `yfinance`, which is free and unofficial. It can occasionally break or rate-limit, and the next scheduled run usually fixes it.
- Earnings come from Yahoo's annual income statements (usually the last 4 years) and analyst estimates. Coverage for some FTSE companies is patchy. Stocks with missing data show as WATCH rather than BUY.
- The 4-year average is a simple mean of daily closes. Prices are adjusted for splits but not for dividends, so the average compares like with like.
- FTSE prices are in pence. The script automatically repairs Yahoo's occasional day quoted in pounds instead of pence.
- The constituent lists are read from Wikipedia on each run and cached in `tickers_cache.csv` as a fallback.
- The index lists are today's members, so the screen has survivorship bias. Companies that fell out of the index aren't shown.
- This is a screening tool, not financial advice.
