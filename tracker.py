#!/usr/bin/env python3
"""
4-Year Average Tracker
======================
Compares every S&P 500 and FTSE 100 stock's latest price with its 4-year
average closing price and writes a self-contained dashboard to docs/index.html.

Strategy (signals are worked out in the page, so the sell threshold can be
changed there without re-running this script):
  * 4-year average = "fair price"
  * price <= 4-year average AND earnings healthy -> BUY
  * price <= 4-year average, earnings weak/unknown -> WATCH (possible value trap)
  * price >= 4-year average * (1 + X%)           -> SELL (X = 10 / 20 / 30, pick in page)
  * anything in between                          -> HOLD

Earnings are healthy when every check that has data passes (see earnings_health):
  1. Profitable       - latest annual EPS > 0
  2. Growing          - latest annual EPS >= EPS three years earlier
  3. No recent drop   - latest annual EPS >= 90% of the year before
  4. Analysts upbeat  - next-year EPS estimate >= trailing 12-month EPS
Checks 1 and 2 must have data; 4 is skipped when there's no analyst estimate.

Usage:
  python tracker.py            # real data from Yahoo Finance (via yfinance)
  python tracker.py --demo     # synthetic data, to preview the page offline
"""
from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "docs"
TEMPLATE = ROOT / "template.html"
TICKER_CACHE = ROOT / "tickers_cache.csv"
WATCHLIST = ROOT / "watchlist.csv"  # optional: extra tickers you want tracked
FUND_CACHE = ROOT / "fundamentals_cache.json"

YEARS = 4
MIN_COVERAGE = 0.9  # need at least 90% of 4 years of history to give a signal
BATCH = 100
# Earnings settings
FUND_SCOPE = 0.05          # fetch earnings for stocks up to 5% above their average
FUND_MAX_AGE_DAYS = 7      # re-download a company's earnings at most once a week
RECENT_DROP_LIMIT = 0.90   # check 3: latest year must be >= 90% of the year before
UA = {"User-Agent": "Mozilla/5.0 (4yr-average-tracker; personal use)"}


# --------------------------------------------------------------------------- #
# 1. Which stocks?
# --------------------------------------------------------------------------- #
def _wiki_tables(url: str) -> list[pd.DataFrame]:
    import requests

    resp = requests.get(url, headers=UA, timeout=30)
    resp.raise_for_status()
    return pd.read_html(io.StringIO(resp.text))


def _find_col(df: pd.DataFrame, *needles: str) -> str | None:
    for col in df.columns:
        name = str(col).lower()
        if any(n in name for n in needles):
            return col
    return None


def get_sp500() -> pd.DataFrame:
    t = _wiki_tables("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
    return pd.DataFrame({
        "ticker": t["Symbol"].astype(str).str.replace(".", "-", regex=False),  # BRK.B -> BRK-B
        "name": t["Security"],
        "sector": t["GICS Sector"],
        "index": "S&P 500",
    })


def get_ftse100() -> pd.DataFrame:
    for t in _wiki_tables("https://en.wikipedia.org/wiki/FTSE_100_Index"):
        tick, comp = _find_col(t, "ticker", "epic"), _find_col(t, "company")
        if tick is None or comp is None or len(t) < 90:
            continue
        sector = _find_col(t, "sector", "industry")

        def yahoo(sym: str) -> str:  # "BT.A" -> "BT-A.L", "RR." -> "RR.L"
            return str(sym).strip().rstrip(".").replace(".", "-") + ".L"

        return pd.DataFrame({
            "ticker": t[tick].map(yahoo),
            "name": t[comp],
            "sector": t[sector] if sector is not None else "",
            "index": "FTSE 100",
        })
    raise RuntimeError("Could not find the FTSE 100 constituents table on Wikipedia")


def get_universe() -> pd.DataFrame:
    try:
        uni = pd.concat([get_sp500(), get_ftse100()], ignore_index=True)
        uni.to_csv(TICKER_CACHE, index=False)
        print(f"Constituents: {len(uni)} (fresh from Wikipedia)")
    except Exception as exc:  # network hiccup / page layout change
        if not TICKER_CACHE.exists():
            raise
        print(f"Wikipedia lookup failed ({exc}); using cached list", file=sys.stderr)
        uni = pd.read_csv(TICKER_CACHE)

    if WATCHLIST.exists():  # columns: ticker,name,sector,index
        extra = pd.read_csv(WATCHLIST).fillna("")
        for col in ("name", "sector"):
            if col not in extra:
                extra[col] = ""
        if "index" not in extra:
            extra["index"] = ""
        extra["index"] = extra["index"].replace("", "Watchlist")
        uni = pd.concat([uni, extra], ignore_index=True)

    return uni.drop_duplicates("ticker").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 2. Prices
# --------------------------------------------------------------------------- #
def download_closes(tickers: list[str]) -> pd.DataFrame:
    """Daily closes (split-adjusted, NOT dividend-adjusted) for ~4 years."""
    import yfinance as yf

    start = (dt.date.today() - dt.timedelta(days=int(365.25 * YEARS) + 21)).isoformat()
    frames = []
    for i in range(0, len(tickers), BATCH):
        batch = tickers[i:i + BATCH]
        print(f"  downloading {i + 1}-{i + len(batch)} of {len(tickers)}")
        for attempt in range(3):
            try:
                df = yf.download(batch, start=start, auto_adjust=False, actions=False,
                                 progress=False, threads=True)
                break
            except Exception as exc:
                print(f"    retry {attempt + 1}: {exc}", file=sys.stderr)
                time.sleep(5 * (attempt + 1))
        else:
            continue
        close = df["Close"]
        if isinstance(close, pd.Series):
            close = close.to_frame(batch[0])
        frames.append(close)
        time.sleep(1)
    if not frames:
        raise RuntimeError("No price data downloaded")
    out = pd.concat(frames, axis=1)
    out.index = pd.to_datetime(out.index).tz_localize(None)
    return out.sort_index()


def clean(s: pd.Series, london: bool) -> pd.Series:
    """Drop bad ticks. Yahoo sometimes quotes a London stock in pounds instead of
    pence for a day, which shows up as a price ~100x too small."""
    s = s[s > 0].dropna()
    if london and len(s) > 20:
        med = s.rolling(21, center=True, min_periods=5).median()
        s = s.where(~(s < med / 50), s * 100)
    return s


# --------------------------------------------------------------------------- #
# 3. Analysis
# --------------------------------------------------------------------------- #
def analyse(closes: pd.DataFrame, uni: pd.DataFrame) -> list[dict]:
    rows = []
    newest = closes.index.max()
    for _, m in uni.iterrows():
        tkr = m["ticker"]
        if tkr not in closes.columns:
            continue
        s = clean(closes[tkr], str(tkr).endswith(".L"))
        if s.empty or (newest - s.index[-1]).days > 10:  # delisted / stale
            continue
        last_date = s.index[-1]
        s4 = s[s.index > last_date - pd.DateOffset(years=YEARS)]
        coverage = (s4.index[-1] - s4.index[0]).days / (365.25 * YEARS)
        price, avg = float(s4.iloc[-1]), float(s4.mean())

        weekly = s4.resample("W-FRI").last().dropna()
        if weekly.index[-1] < last_date:  # make sure the last point is today's price
            weekly.loc[last_date] = price

        rows.append({
            "t": tkr,
            "n": str(m["name"]),
            "s": "" if pd.isna(m["sector"]) else str(m["sector"]),
            "i": str(m["index"]),
            "c": "p" if str(tkr).endswith(".L") else "$",
            "p": round(price, 2),
            "a": round(avg, 2),
            "lo": round(float(s4.min()), 2),
            "hi": round(float(s4.max()), 2),
            "cov": round(min(coverage, 1.0), 3),
            "d": last_date.strftime("%Y-%m-%d"),
            "h0": weekly.index[0].strftime("%Y-%m-%d"),
            "h": [round(float(v), 2) for v in weekly.values],
        })
    return rows


# --------------------------------------------------------------------------- #
# 3b. Earnings health (only for stocks at, below or near their average)
# --------------------------------------------------------------------------- #
def fetch_fundamentals(tkr: str) -> dict:
    import yfinance as yf

    tk = yf.Ticker(tkr)
    stmt = tk.income_stmt
    eps, kind, years = [], None, []
    if stmt is not None and not stmt.empty:
        for row, label in (("Diluted EPS", "EPS"), ("Basic EPS", "EPS"), ("Net Income", "Net income")):
            if row in stmt.index:
                s = pd.to_numeric(stmt.loc[row], errors="coerce").dropna().sort_index()
                if len(s) >= 2:
                    eps, kind = [float(v) for v in s.values][-5:], label
                    years = [pd.Timestamp(d).strftime("%Y") for d in s.index][-5:]
                    break
    try:
        info = tk.info or {}
    except Exception:
        info = {}
    num = lambda v: float(v) if isinstance(v, (int, float)) and np.isfinite(v) else None
    return {
        "fetched": dt.date.today().isoformat(),
        "kind": kind, "years": years, "eps": eps,
        "ttm": num(info.get("trailingEps")), "fwd": num(info.get("forwardEps")),
    }


def load_fundamentals(tickers: list[str]) -> dict:
    cache = json.loads(FUND_CACHE.read_text()) if FUND_CACHE.exists() else {}
    today = dt.date.today()
    stale = [t for t in tickers
             if t not in cache
             or (today - dt.date.fromisoformat(cache[t]["fetched"])).days >= FUND_MAX_AGE_DAYS]
    print(f"Earnings: {len(tickers)} candidates, {len(stale)} to download, rest cached")
    for n, tkr in enumerate(stale, 1):
        for attempt in range(3):
            try:
                cache[tkr] = fetch_fundamentals(tkr)
                break
            except Exception as exc:
                print(f"    {tkr} retry {attempt + 1}: {exc}", file=sys.stderr)
                time.sleep(4 * (attempt + 1))
        if n % 25 == 0:
            print(f"  earnings {n}/{len(stale)}")
            FUND_CACHE.write_text(json.dumps(cache))
        time.sleep(0.4)
    FUND_CACHE.write_text(json.dumps(cache))
    return cache


def _chg(new: float, old: float) -> float | None:
    return None if old is None or old <= 0 else new / old - 1


def earnings_health(f: dict | None, price_vs_avg: float) -> dict:
    """Returns {"ok": bool|None, "checks": [[label, pass(True/False/None), detail], ...], "pe": str|None}"""
    if not f or not f.get("eps"):
        return {"ok": None, "checks": [], "pe": None}
    eps, years, kind = f["eps"], f["years"], f["kind"]
    latest = eps[-1]
    pct = lambda x: f"{x:+.0%}"
    checks = []

    # 1. Profitable
    checks.append(["Profitable", latest > 0,
                   f"{kind} {latest:,.2f} in {years[-1]}" + ("" if latest > 0 else " (a loss)")])

    # 2. Growing over ~3 years
    if len(eps) >= 4:
        base, by = eps[-4], years[-4]
    elif len(eps) >= 3:
        base, by = eps[0], years[0]
    else:
        base = by = None
    if base is None:
        checks.append(["Growing over 3 years", None, "Not enough annual history"])
    else:
        g = _chg(latest, base)
        ok = latest >= base and latest > 0
        checks.append(["Growing over 3 years", ok,
                       f"{pct(g)} since {by}" if g is not None else f"{by} was a loss"])

    # 3. No recent drop
    prev, py = eps[-2], years[-2]
    g = _chg(latest, prev)
    checks.append(["No recent drop", latest >= RECENT_DROP_LIMIT * prev if prev > 0 else latest > 0,
                   f"{pct(g)} vs {py}" if g is not None else f"{py} was a loss"])

    # 4. Analysts expect growth
    ttm, fwd = f.get("ttm"), f.get("fwd")
    if ttm is None or fwd is None or ttm <= 0:
        checks.append(["Analysts expect growth", None, "No analyst estimate available"])
    else:
        checks.append(["Analysts expect growth", fwd >= ttm, f"Next-year estimate {pct(fwd / ttm - 1)}"])

    must_have = checks[0][1] is not None and checks[1][1] is not None
    ok = must_have and all(c[1] is not False for c in checks)

    # Valuation context (not a gate): how today's P/E compares with its 4-year norm
    pe = None
    recent = [e for e in eps[-4:]]
    avg_eps = sum(recent) / len(recent)
    if latest > 0 and avg_eps > 0:
        pe_rel = (1 + price_vs_avg) / (latest / avg_eps) - 1
        pe = f"P/E is {abs(pe_rel):.0%} {'below' if pe_rel < 0 else 'above'} its 4-year norm"
    return {"ok": ok, "checks": checks, "pe": pe}

# --------------------------------------------------------------------------- #
# 4. Demo data (no network needed)
# --------------------------------------------------------------------------- #
def demo_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(7)
    days = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=int(252 * YEARS) + 15)
    sectors = ["Technology", "Financials", "Health Care", "Energy", "Industrials", "Consumer"]
    meta, cols = [], {}
    for k in range(48):
        london = k % 3 == 0
        tkr = f"DEMO{k + 1:02d}" + (".L" if london else "")
        drift = rng.normal(0.0002, 0.0006)
        vol = rng.uniform(0.010, 0.025)
        path = np.exp(np.cumsum(rng.normal(drift, vol, len(days))))
        cols[tkr] = (rng.uniform(800, 4000) if london else rng.uniform(20, 400)) * path
        meta.append({"ticker": tkr, "name": f"Example Company {k + 1}",
                     "sector": sectors[k % len(sectors)],
                     "index": "FTSE 100" if london else "S&P 500"})
    closes = pd.DataFrame(cols, index=days)
    closes.iloc[: len(days) // 3, 5] = np.nan  # one recent listing -> "not enough history"
    return closes, pd.DataFrame(meta)


def demo_fundamentals(tickers: list[str]) -> dict:
    rng = np.random.default_rng(11)
    yr = dt.date.today().year
    out = {}
    for k, t in enumerate(tickers):
        if k % 9 == 4:
            out[t] = {"eps": [], "years": [], "kind": None, "ttm": None, "fwd": None}  # no data
            continue
        growth = rng.normal(0.06, 0.15)
        eps = list(np.round(rng.uniform(1, 8) * np.cumprod(1 + rng.normal(growth, 0.12, 4)), 2))
        if k % 7 == 3:
            eps[-1] = -abs(eps[-1]) * 0.3  # a loss-making year
        ttm = eps[-1] * (1 + rng.normal(0.02, 0.05))
        out[t] = {"eps": [float(e) for e in eps], "years": [str(yr - 4 + i) for i in range(4)],
                  "kind": "EPS", "ttm": float(ttm),
                  "fwd": None if k % 5 == 0 else float(ttm * (1 + rng.normal(0.05, 0.1)))}
    return out


# --------------------------------------------------------------------------- #
# 5. Output
# --------------------------------------------------------------------------- #
def write_outputs(rows: list[dict], demo: bool) -> Path:
    payload = {
        "generated": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "years": YEARS,
        "minCoverage": MIN_COVERAGE,
        "demo": demo,
        "stocks": rows,
    }
    OUT_DIR.mkdir(exist_ok=True)
    blob = json.dumps(payload, separators=(",", ":"))
    (OUT_DIR / "data.json").write_text(blob)
    html = TEMPLATE.read_text().replace("/*__DATA__*/null", blob.replace("</", "<\\/"))
    out = OUT_DIR / "index.html"
    out.write_text(html)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", action="store_true", help="use synthetic prices (no internet needed)")
    args = ap.parse_args()

    if args.demo:
        closes, uni = demo_data()
    else:
        uni = get_universe()
        closes = download_closes(uni["ticker"].tolist())

    rows = analyse(closes, uni)

    near = [r["t"] for r in rows if r["p"] / r["a"] - 1 <= FUND_SCOPE]
    funds = demo_fundamentals(near) if args.demo else load_fundamentals(near)
    for r in rows:
        if r["t"] in funds:
            r["e"] = earnings_health(funds[r["t"]], r["p"] / r["a"] - 1)

    out = write_outputs(rows, args.demo)
    cheap = [r for r in rows if r["p"] <= r["a"] and r["cov"] >= MIN_COVERAGE]
    buys = sum(1 for r in cheap if r.get("e", {}).get("ok"))
    print(f"Analysed {len(rows)} stocks; {len(cheap)} at or below their {YEARS}-year average, "
          f"{buys} of them with healthy earnings (BUY), {len(cheap) - buys} WATCH.")
    print(f"Dashboard written to {out}")


if __name__ == "__main__":
    main()
