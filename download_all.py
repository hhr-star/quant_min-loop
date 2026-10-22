"""全市场数据下载（新浪源）。

为什么不用东财：全市场需要 5000+ 次请求，东财会限流，
单只耗时 12 秒且只返回最近 640 行，无法覆盖 2019 年以来的区间。
新浪单只约 0.2 秒、返回 2000 行，是唯一可行的高速源。

代价：新浪不提供成交额和换手率，因此：
    - 成交额用 收盘价 × 成交量 估算（只用于流动性过滤，足够）
    - 换手率缺失，全市场版本只用「动量 + 低波」两个因子

用法：
    python download_all.py
    python download_all.py --workers 8
    python download_all.py --force
"""
import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pandas as pd

import config
import download as dl

DATA_DIR = "data_all"
SINA_KLINE = ("https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
              "CN_MarketData.getKLineData")
SINA_LIST = ("https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/"
             "Market_Center.getHQNodeData")


def sina_prefix(symbol: str) -> str:
    return "sh" if symbol.startswith(("6", "9", "5")) else "sz"


def fetch_one_sina(symbol: str, start: str, end: str) -> pd.DataFrame:
    """新浪日线：2000 根 K 线，覆盖约 8 年。"""
    params = {"symbol": sina_prefix(symbol) + symbol, "scale": 240,
              "ma": "no", "datalen": 2000}
    last = None
    for k in range(4):
        try:
            r = dl._SESSION.get(SINA_KLINE, params=params, timeout=20)
            data = r.json()
            break
        except Exception as e:                       # noqa: BLE001
            last = e
            data = None
            time.sleep(0.5 * (2 ** k))
    if not isinstance(data, list) or not data:
        if last:
            raise last
        return pd.DataFrame()

    recs = []
    for d in data:
        recs.append({
            "date": d["day"], "open": float(d["open"]), "close": float(d["close"]),
            "high": float(d["high"]), "low": float(d["low"]),
            "volume": float(d["volume"]),
        })
    df = pd.DataFrame(recs)
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))]
    if len(df) == 0:
        return df
    df["symbol"] = symbol
    df["amount"] = df["close"] * df["volume"] / 1e4      # 元 -> 万元（估算）
    df["pct_chg"] = df["close"].pct_change(fill_method=None) * 100
    df["turnover"] = float("nan")
    return df.sort_values("date").reset_index(drop=True)


def fetch_universe_all() -> pd.DataFrame:
    """新浪全 A 列表：沪深分开拉，逐页重试。"""
    path = os.path.join(DATA_DIR, "universe.csv")
    if os.path.exists(path):
        return pd.read_csv(path, dtype={"symbol": str})

    rows = []
    for node in ["sh_a", "sz_a"]:
        got = 0
        for page in range(1, 80):
            data = None
            for _ in range(4):
                try:
                    data = dl._get(SINA_LIST, {"page": page, "num": 100,
                                               "sort": "symbol", "asc": 1, "node": node})
                    if isinstance(data, list) and data:
                        break
                except Exception:                    # noqa: BLE001
                    pass
                data = None
                time.sleep(1.5)
            if not data:
                break
            for it in data:
                rows.append({"symbol": str(it.get("code", "")).zfill(6),
                             "name": it.get("name", "")})
            got += len(data)
            time.sleep(0.2)
        print(f"  [universe] {node}: {got} 只", flush=True)

    df = pd.DataFrame(rows).drop_duplicates("symbol")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[universe] 全市场 {len(df)} 只")
    return df


def main(force=False, limit=None, workers=8):
    os.makedirs(DATA_DIR, exist_ok=True)
    universe = fetch_universe_all()
    symbols = universe["symbol"].tolist()
    if limit:
        symbols = symbols[:limit]

    panel_path = os.path.join(DATA_DIR, "panel.csv.gz")
    part_path = panel_path + ".part"
    existing, done = pd.DataFrame(), set()
    resume = part_path if os.path.exists(part_path) else panel_path
    if os.path.exists(resume) and not force:
        existing = pd.read_csv(resume, parse_dates=["date"], dtype={"symbol": str},
                               compression="gzip")
        done = set(existing["symbol"].unique())
        print(f"[cache] 续传，已有 {len(done)} 只")

    todo = [s for s in symbols if s not in done]
    print(f"[run] 待下载 {len(todo)} 只，线程 {workers}", flush=True)

    frames, ok, bad = [], 0, []
    t0 = time.time()
    BATCH = 400
    for b0 in range(0, len(todo), BATCH):
        batch = todo[b0:b0 + BATCH]
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(fetch_one_sina, s, config.START_DATE, config.END_DATE): s
                    for s in batch}
            for fut in as_completed(futs):
                sym = futs[fut]
                try:
                    d = fut.result()
                    if len(d):
                        frames.append(d)
                        ok += 1
                    else:
                        bad.append(sym)
                except Exception:                    # noqa: BLE001
                    bad.append(sym)
        n = min(b0 + BATCH, len(todo))
        print(f"[{n}/{len(todo)}] 成功 {ok} 失败 {len(bad)} 用时 {time.time()-t0:.0f}s",
              flush=True)
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(
                part_path, index=False, compression="gzip")

    if frames:
        new = pd.concat(frames, ignore_index=True)
        panel = pd.concat([existing, new], ignore_index=True) if len(existing) else new
    else:
        panel = existing

    if len(panel) == 0:
        print("没有拿到数据", file=sys.stderr)
        sys.exit(1)

    panel = panel.drop_duplicates(["date", "symbol"]).sort_values(["date", "symbol"])
    panel.to_csv(panel_path, index=False, compression="gzip")
    if os.path.exists(part_path):
        os.remove(part_path)
    print(f"[panel] {panel['symbol'].nunique()} 只 / {len(panel)} 行 / "
          f"{panel['date'].min().date()} ~ {panel['date'].max().date()}")
    if bad:
        print(f"[warn] 失败 {len(bad)} 只", file=sys.stderr)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    main(force=a.force, limit=a.limit, workers=a.workers)
