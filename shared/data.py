"""Fetching public datasets, once, with the provenance recorded.

Every dataset in this repo is public and downloadable without an account. Each fetch
records where the file came from and when, because a result computed on "some CSV I had"
is not reproducible and should not be believed — including by the person who computed it.

Files land in `data/raw/` (git-ignored). Delete the directory and everything re-fetches.
"""

from __future__ import annotations

import hashlib
import io
import json
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
MANIFEST = ROOT / "data" / "sources.json"

USER_AGENT = "machine-learning-portfolio/0.1 (public dataset fetch)"
TIMEOUT = 120


class FetchFailed(RuntimeError):
    """The source was unreachable. Stated loudly rather than silently falling back."""


@dataclass(frozen=True)
class Source:
    name: str
    url: str
    note: str
    licence: str


def _download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})  # noqa: S310
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:  # noqa: S310
            return response.read()
    except (urllib.error.URLError, TimeoutError) as exc:
        raise FetchFailed(f"could not reach {url}: {exc}") from exc


def _record(source: Source, path: Path, payload: bytes) -> None:
    """Append provenance so any number in this repo can be traced to a byte range."""
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    entries = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    entries[source.name] = {
        "url": source.url,
        "note": source.note,
        "licence": source.licence,
        "fetched": datetime.now(UTC).isoformat(timespec="seconds"),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "path": str(path.relative_to(ROOT)),
    }
    MANIFEST.write_text(json.dumps(entries, indent=2, sort_keys=True))


def fetch(source: Source, filename: str | None = None) -> Path:
    """Download once, cache, record provenance, return the local path."""
    RAW.mkdir(parents=True, exist_ok=True)
    target = RAW / (filename or f"{source.name}{Path(source.url).suffix or '.csv'}")

    if target.exists() and target.stat().st_size > 0:
        return target

    print(f"  fetching {source.name} ...", flush=True)
    payload = _download(source.url)
    target.write_bytes(payload)
    _record(source, target, payload)
    print(f"  wrote {target.name} ({len(payload):,} bytes)")
    return target


def fetch_csv(source: Source, filename: str | None = None, **read_csv) -> pd.DataFrame:
    """Fetch and parse in one step."""
    return pd.read_csv(fetch(source, filename), **read_csv)


def fetch_zip_member(source: Source, member: str, *, filename: str | None = None) -> pd.DataFrame:
    """Pull a single CSV out of a zip archive without unpacking the whole thing."""
    path = fetch(source, filename or f"{source.name}.zip")
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        match = next((n for n in names if n.endswith(member)), None)
        if match is None:
            raise FetchFailed(f"{member} not in {path.name}; archive holds {names[:8]}")
        with archive.open(match) as handle:
            return pd.read_csv(io.BytesIO(handle.read()))


# --- the catalogue --------------------------------------------------------
#
# Every entry below was reachable and returned real data on 2026-09-11. That date is
# recorded because public data links rot, and a catalogue that silently 404s is worse
# than no catalogue: the pipeline keeps running on a stale cache and nobody notices.
#
# Two sources were dropped during setup rather than left in as aspiration:
#   Stooq and FRED  - now behind a JavaScript browser challenge, no plain CSV
#   MovieLens       - files.grouplens.org refused the connection
# `make sources` re-checks every URL and reports which have gone.

STOCK_DAILY = Source(
    name="stock-daily-5tickers",
    url="https://raw.githubusercontent.com/plotly/datasets/master/stockdata.csv",
    note="Daily closes for MSFT, IBM, SBUX, AAPL and GSPC (the S&P 500 index), "
    "2,306 trading days. Used for the backtest-overfitting search.",
    licence="MIT (plotly/datasets)",
)

VIX_DAILY = Source(
    name="cboe-vix-daily",
    url="https://raw.githubusercontent.com/datasets/finance-vix/main/data/vix-daily.csv",
    note="CBOE VIX daily OHLC since 1990, 9,269 rows. The volatility-regime series.",
    licence="Public domain (CBOE publishes VIX history freely)",
)

SHILLER_SP500 = Source(
    name="shiller-sp500-monthly",
    url="https://raw.githubusercontent.com/datasets/s-and-p-500/main/data/data.csv",
    note="Robert Shiller's long-run S&P 500 series, monthly since 1871: price, "
    "dividends, earnings, CPI, long rate and CAPE.",
    licence="Public (Shiller, Yale) via datasets/s-and-p-500",
)

FOOTBALL_E0 = Source(
    name="football-data-e0-2324",
    url="https://www.football-data.co.uk/mmz4281/2324/E0.csv",
    note="English Premier League 2023/24: every result plus closing odds from several "
    "bookmakers. The odds are the benchmark any model has to beat.",
    licence="Free to use with attribution (football-data.co.uk)",
)

AI4I_MAINTENANCE = Source(
    name="ai4i-2020-maintenance",
    url="https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip",
    note="UCI AI4I 2020: 10,000 machine cycles, five labelled failure modes, 3.4% "
    "failure rate. Imbalanced by construction, like the real thing.",
    licence="CC BY 4.0 (UCI ML Repository)",
)

ONLINE_RETAIL = Source(
    name="uci-online-retail-ii",
    url="https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip",
    note="UCI Online Retail II: ~1M invoice lines from a UK gift wholesaler, 2009-2011. "
    "Real transaction data with returns, cancellations and a long tail of customers.",
    licence="CC BY 4.0 (UCI ML Repository)",
)

BANK_MARKETING = Source(
    name="uci-bank-marketing",
    url="https://archive.ics.uci.edu/static/public/222/bank+marketing.zip",
    note="UCI Bank Marketing: 45,211 phone contacts from a Portuguese bank campaign, "
    "with the subscription outcome. 11.7% positive rate.",
    licence="CC BY 4.0 (UCI ML Repository)",
)

CATALOGUE = {
    s.name: s
    for s in (
        STOCK_DAILY,
        VIX_DAILY,
        SHILLER_SP500,
        FOOTBALL_E0,
        AI4I_MAINTENANCE,
        ONLINE_RETAIL,
        BANK_MARKETING,
    )
}


def status(*, check_remote: bool = False) -> None:
    """Print what is cached. With check_remote, also re-verify every URL still serves."""
    print(f"{'dataset':<28} {'cached':<8} {'remote':<8} note")
    for name, source in CATALOGUE.items():
        hit = any(RAW.glob(f"{name}.*")) if RAW.exists() else False
        remote = ""
        if check_remote:
            try:
                _download(source.url)
                remote = "ok"
            except FetchFailed:
                remote = "GONE"
        print(f"{name:<28} {'yes' if hit else 'no':<8} {remote:<8} {source.note[:56]}")


if __name__ == "__main__":
    import sys

    status(check_remote="--check" in sys.argv)


def online_retail() -> pd.DataFrame:
    """Online Retail II, cleaned, with the cleaning decisions stated.

    UCI ships this as a 45 MB Excel workbook with two sheets, which takes about a minute
    to parse. It is converted to a CSV on first use and read from there afterwards.

    Four cleaning rules, each of which changes the answer and so is written down:

      1. **Cancellations are real.** Invoice numbers beginning with C are returns and carry
         negative quantities. Dropping them inflates revenue; keeping them means a customer
         who bought and returned nets to roughly zero, which is the truth.
      2. **Rows without a Customer ID cannot be attributed.** About 22% of lines. They are
         genuine sales but cannot join to a customer, so they are excluded from anything
         per-customer and that exclusion is reported rather than assumed harmless.
      3. **Non-positive prices are not sales.** Zero and negative unit prices are
         adjustments, debts written off, and stock corrections.
      4. **Postage and fees are not products.** Stock codes such as POST, DOT, M, BANK
         CHARGES and ADJUST are charges; leaving them in makes "postage" a best-seller.
    """
    cache = RAW / "online-retail-ii.csv"
    if not cache.exists():
        path = fetch(ONLINE_RETAIL, "uci-online-retail-ii.zip")
        print("  converting the Excel workbook once (this takes a minute) ...", flush=True)
        with zipfile.ZipFile(path) as archive:
            member = next(n for n in archive.namelist() if n.endswith(".xlsx"))
            payload = io.BytesIO(archive.read(member))
        sheets = pd.read_excel(payload, sheet_name=None, engine="openpyxl")
        frame = pd.concat(sheets.values(), ignore_index=True)
        frame.columns = [str(c).strip().replace(" ", "") for c in frame.columns]
        frame.to_csv(cache, index=False)
        print(f"  cached {cache.name} ({cache.stat().st_size:,} bytes, {len(frame):,} rows)")

    frame = pd.read_csv(cache, parse_dates=["InvoiceDate"], low_memory=False)
    frame["Invoice"] = frame["Invoice"].astype(str)
    frame["StockCode"] = frame["StockCode"].astype(str).str.upper().str.strip()

    not_products = {
        "POST",
        "DOT",
        "M",
        "S",
        "BANK CHARGES",
        "AMAZONFEE",
        "ADJUST",
        "ADJUST2",
        "TEST001",
        "TEST002",
        "B",
        "CRUK",
        "PADS",
        "GIFT",
    }
    frame = frame[~frame["StockCode"].isin(not_products)]
    frame = frame[frame["Price"] > 0]

    frame["is_return"] = frame["Invoice"].str.startswith("C")
    frame["revenue"] = frame["Quantity"] * frame["Price"]
    return frame.reset_index(drop=True)
