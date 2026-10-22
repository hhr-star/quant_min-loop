"""数据下载：直接调东方财富公开接口，无第三方数据依赖。

用法：
    python download.py                 # 增量下载
    python download.py --force         # 强制重下
    python download.py --limit 30      # 只下前 30 只，先验证流程

产物：
    data/universe.csv    成分股清单（symbol, name）
    data/panel.csv.gz    个股日线长表
    data/index.csv       基准指数日线
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

import config

DATA_DIR = config.DATA_DIR
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Referer": "https://quote.eastmoney.com/",
    "Accept": "*/*",
    "Connection": "keep-alive",
}

_SESSION = requests.Session()
_SESSION.headers.update(HEADERS)


def _get(url: str, params: dict, retries: int = 4) -> dict:
    """带指数退避的 GET，返回 JSON。东财偶发断连，必须重试。"""
    last = None
    for k in range(retries):
        try:
            r = _SESSION.get(url, params=params, timeout=20)
            r.raise_for_status()
            return r.json()
        except Exception as e:          # noqa: BLE001
            last = e
            time.sleep(0.8 * (2 ** k))  # 0.8s, 1.6s, 3.2s ...
    raise last

# 东财 kline 字段顺序：日期,开,收,高,低,成交量(手),成交额(元),振幅,涨跌幅,涨跌额,换手率
KLINE_FIELDS = "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61"


def secid(symbol: str) -> str:
    """东财 secid 规则：沪市 1.xxxxxx，深市 0.xxxxxx。"""
    return ("1." if symbol.startswith(("6", "9", "5")) else "0.") + symbol


def fetch_universe() -> pd.DataFrame:
    """拉沪深300 成分股全部分页。"""
    path = os.path.join(DATA_DIR, "universe.csv")
    if os.path.exists(path):
        return pd.read_csv(path, dtype={"symbol": str})

    rows = []
    for pn in range(1, 6):
        url = "https://push2.eastmoney.com/api/qt/clist/get"
        params = {
            "pn": pn, "pz": 100, "po": 1, "np": 1, "fltt": 2, "invt": 2,
            "fid": "f12", "fs": "b:BK0500", "fields": "f12,f14",
        }
        diff = (_get(url, params).get("data") or {}).get("diff") or []
        if not diff:
            break
        for it in diff:
            rows.append({"symbol": str(it["f12"]).zfill(6), "name": it.get("f14", "")})

    df = pd.DataFrame(rows).drop_duplicates("symbol")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[universe] {len(df)} 只成分股")
    return df


def fetch_one(symbol: str, start: str, end: str) -> pd.DataFrame:
    """拉单只股票前复权日线。

    东财优先（字段全，含换手率）；科创板 688 被东财拒绝，
    以及东财偶发失败时，退回腾讯接口（无换手率字段）。
    """
    if not symbol.startswith("688"):
        try:
            return fetch_one_em(symbol, start, end)
        except Exception:                      # noqa: BLE001
            pass
    return fetch_one_tx(symbol, start, end)


def fetch_one_tx(symbol: str, start: str, end: str) -> pd.DataFrame:
    """腾讯日线兜底：无成交额/换手率字段，成交额按 收盘×成交量 估算。"""
    prefix = "sh" if symbol.startswith(("6", "9", "5")) else "sz"
    url = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
    params = {"param": f"{prefix}{symbol},day,{start},{end},640,qfq"}
    node = (_get(url, params).get("data") or {}).get(f"{prefix}{symbol}") or {}
    rows = node.get("qfqday") or node.get("day") or []
    if not rows:
        return pd.DataFrame()

    recs = []
    for p in rows:
        recs.append({
            "date": p[0], "open": float(p[1]), "close": float(p[2]),
            "high": float(p[3]), "low": float(p[4]), "volume": float(p[5]),
        })
    df = pd.DataFrame(recs)
    df["date"] = pd.to_datetime(df["date"])
    df["symbol"] = symbol
    df["amount"] = df["close"] * df["volume"] / 1e4      # 元 -> 万元
    df["pct_chg"] = df["close"].pct_change(fill_method=None) * 100
    df["turnover"] = float("nan")                        # 该源不提供
    return df.sort_values("date").reset_index(drop=True)


def fetch_one_em(symbol: str, start: str, end: str) -> pd.DataFrame:
    url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
    params = {
        "secid": secid(symbol),
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": KLINE_FIELDS,
        "klt": 101,          # 日线
        "fqt": 1,            # 前复权
        "beg": start.replace("-", ""),
        "end": end.replace("-", ""),
        "lmt": 100000,
    }
    data = _get(url, params).get("data") or {}
    klines = data.get("klines") or []
    if not klines:
        return pd.DataFrame()

    recs = []
    for line in klines:
        p = line.split(",")
        recs.append({
            "date": p[0], "open": float(p[1]), "close": float(p[2]),
            "high": float(p[3]), "low": float(p[4]),
            "volume": float(p[5]), "amount": float(p[6]) / 1e4,  # 元 -> 万元
            "pct_chg": float(p[8]), "turnover": float(p[10]),
        })
    df = pd.DataFrame(recs)
    df["date"] = pd.to_datetime(df["date"])
    df["symbol"] = symbol
    return df.sort_values("date").reset_index(drop=True)


def fetch_index():
    """沪深300 指数日线（走新浪源，覆盖面足够）。"""
    url = ("https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
           "CN_MarketData.getKLineData")
    params = {"symbol": "sh000300", "scale": 240, "ma": "no", "datalen": 2000}
    rows = _get(url, params)
    recs = [{"date": r["day"], "close": float(r["close"])} for r in rows]
    df = pd.DataFrame(recs)
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= config.START_DATE) & (df["date"] <= config.END_DATE)]
    path = os.path.join(DATA_DIR, "index.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[index] {len(df)} 行 {df['date'].min().date()} ~ {df['date'].max().date()}")


def download_all(force=False, limit=None, workers=6):
    os.makedirs(DATA_DIR, exist_ok=True)
    universe = fetch_universe()
    symbols = universe["symbol"].tolist()
    if limit:
        symbols = symbols[:limit]

    panel_path = os.path.join(DATA_DIR, "panel.csv.gz")
    existing, done = pd.DataFrame(), set()
    if os.path.exists(panel_path) and not force:
        existing = pd.read_csv(panel_path, parse_dates=["date"], dtype={"symbol": str})
        done = set(existing["symbol"].unique())
        print(f"[cache] 已有 {len(done)} 只")

    todo = [s for s in symbols if s not in done]
    print(f"[run] 待下载 {len(todo)} 只，线程数 {workers}")

    frames, ok, bad = [], 0, []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch_one, s, config.START_DATE, config.END_DATE): s for s in todo}
        for n, fut in enumerate(as_completed(futs), 1):
            sym = futs[fut]
            try:
                d = fut.result()
                if len(d):
                    frames.append(d)
                    ok += 1
                else:
                    bad.append(sym)
            except Exception as e:
                bad.append(sym)
                print(f"  ! {sym} {type(e).__name__}", file=sys.stderr)
            if n % 25 == 0 or n == len(todo):
                el = time.time() - t0
                print(f"[{n}/{len(todo)}] 成功 {ok} 失败 {len(bad)} 用时 {el:.0f}s", flush=True)

    if frames:
        new = pd.concat(frames, ignore_index=True)
        panel = pd.concat([existing, new], ignore_index=True) if len(existing) else new
    else:
        panel = existing

    if len(panel) == 0:
        print("没有拿到任何数据", file=sys.stderr)
        sys.exit(1)

    panel = panel.drop_duplicates(["date", "symbol"]).sort_values(["date", "symbol"])
    panel.to_csv(panel_path, index=False, compression="gzip")
    print(f"[panel] {panel['symbol'].nunique()} 只 / {len(panel)} 行 / "
          f"{panel['date'].min().date()} ~ {panel['date'].max().date()}")

    fetch_index()
    if bad:
        print(f"[warn] 失败 {len(bad)} 只：{bad[:20]}", file=sys.stderr)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--workers", type=int, default=6)
    a = p.parse_args()
    download_all(force=a.force, limit=a.limit, workers=a.workers)
