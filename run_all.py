"""通用回测入口：可在任意股票池上跑同一套策略。

全市场数据（新浪源）没有换手率字段，所以这里统一用「动量 + 低波」两个因子，
保证沪深300 池与全 A 池的对比只差「股票池」一个变量。

用法：
    # 沪深300 池（东财数据，含换手率但本脚本不使用）
    python run_all.py --panel data/panel.csv.gz --universe data/universe.csv \
        --out reports_index_2f --min-amount 2000

    # 全 A 池
    python run_all.py --panel data_all/panel.csv.gz --universe data_all/universe.csv \
        --out reports_all --min-amount 1000
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

import backtest
import config
import factors
import metrics
import run as base


def load_data(panel_path, universe_path):
    panel = pd.read_csv(panel_path, parse_dates=["date"], dtype={"symbol": str})
    universe = pd.read_csv(universe_path, dtype={"symbol": str})
    # 只保留股票池内的标的（panel 可能是全 A，universe 可能是沪深300）
    pool = set(universe["symbol"])
    before = panel["symbol"].nunique()
    panel = panel[panel["symbol"].isin(pool)].copy()
    print(f"[pool] {before} -> {panel['symbol'].nunique()} 只")
    index_df = pd.read_csv(os.path.join("data", "index.csv"), parse_dates=["date"])
    return panel, index_df, universe


def build_mask(panel, universe, min_amount):
    """可交易掩码：流动性 + 上市满 250 个交易日 + 非 ST。

    上市天数必须按「每只股票自己的首个交易日」算：
    数据区间起点就存在的股票视为早已上市，直接放行；
    区间中途才出现的才当新股，等 250 个交易日。
    否则区间开头的 250 天会把所有股票都误判成新股。
    """
    amount = factors.pivot(panel, "amount")
    close = factors.pivot(panel, "close")

    avg_amount = amount.rolling(20, min_periods=10).mean()
    mask = (avg_amount >= min_amount) & close.notna()

    arr = close.notna().values
    n = arr.shape[0]
    first_idx = arr.argmax(axis=0)
    first_idx[~arr.any(axis=0)] = 0
    late = first_idx > 0
    allow = np.ones_like(arr, dtype=bool)
    if late.any():
        allow[:, late] = np.arange(n)[:, None] >= (first_idx[late] + 250)[None, :]
    mask &= pd.DataFrame(allow, index=mask.index, columns=mask.columns)

    if config.EXCLUDE_ST:
        st = set(universe.loc[universe["name"].astype(str).str.contains("ST", na=False), "symbol"])
        hit = [c for c in mask.columns if c in st]
        if hit:
            mask[hit] = False
            print(f"[mask] 剔除 ST {len(hit)} 只")
    return mask


def build_score_2f(fac, reverse=False):
    """动量 + 低波，两因子等权。

    reverse=True 时动量取反，变成「反转 + 低波」——
    这是对 A 股动量 IC 为负的直接检验。
    """
    sign = -1 if reverse else 1
    z_mom = factors.cross_section_zscore(fac["momentum"] * sign)
    z_vol = factors.cross_section_zscore(-fac["volatility"])
    score = z_mom.fillna(0) + z_vol.fillna(0)
    valid = z_mom.notna() | z_vol.notna()
    return score.where(valid)


def main(panel_path, universe_path, out_dir, min_amount, reverse=False):
    os.makedirs(out_dir, exist_ok=True)
    panel, index_df, universe = load_data(panel_path, universe_path)
    print(f"[data] {panel['symbol'].nunique()} 只 / {len(panel)} 行")

    fac = factors.compute_factors(panel)
    score = build_score_2f(fac, reverse=reverse)
    mask = build_mask(panel, universe, min_amount)

    res = backtest.run_backtest(panel, score, mask)
    nav = res["nav"]
    print(f"[backtest] 净值 {nav.iloc[-1]:.3f}，调仓 {len(res['holdings'])} 次")

    bench = index_df.set_index("date")["close"].reindex(nav.index).ffill()
    bench_nav = bench / bench.dropna().iloc[0]

    split = pd.Timestamp(config.IN_SAMPLE_END)
    segs = {"样本内": nav[nav.index <= split], "样本外": nav[nav.index > split], "全期": nav}
    rows = {}
    for name, s in segs.items():
        if len(s) < 2:
            continue
        st = metrics.summarize(s)
        tov = res["turnover"][res["turnover"].index.isin(s.index)]
        st["年均单边换手"] = tov.mean() * metrics.TRADING_DAYS / 2
        rows[name] = st
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(out_dir, "summary.csv"), encoding="utf-8-sig")

    yearly = metrics.yearly_returns(nav).to_frame("策略")
    yearly.index.name = "年份"
    yearly["沪深300"] = metrics.yearly_returns(bench_nav.dropna())
    yearly.to_csv(os.path.join(out_dir, "yearly.csv"), encoding="utf-8-sig")

    base.REPORT_DIR = out_dir
    base.plot_nav(nav, bench_nav)
    base.plot_monthly(nav)
    base.plot_yearly(yearly)

    lines = ["# 两因子回测（动量 + 低波）\n",
             f"- 股票池：{panel['symbol'].nunique()} 只",
             f"- 流动性门槛：20 日均成交额 ≥ {min_amount} 万元",
             "- 因子：20 日动量 + 60 日低波，等权合成\n",
             "## 绩效\n", base.md_table(summary),
             "\n## 分年度\n", base.md_table(yearly, index_name="年份", pct_mode="all"),
             "\n## 图表\n", "![净值](nav.png)\n", "![月度](monthly.png)\n"]
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[report] {out_dir}/")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--panel", default="data_all/panel.csv.gz")
    p.add_argument("--universe", default="data_all/universe.csv")
    p.add_argument("--out", default="reports_all")
    p.add_argument("--min-amount", type=float, default=1000)
    p.add_argument("--reverse", action="store_true", help="动量取反，做反转策略")
    a = p.parse_args()
    main(a.panel, a.universe, a.out, a.min_amount, reverse=a.reverse)
