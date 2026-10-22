"""因子有效性分析：Rank IC、分层回测、分年度稳定性。

回答第 2 周的问题：
    动量因子在哪些年份有效？为什么在全 A 池上失效？
    是因子本身衰减，还是被「大盘 vs 全市场」的风格切换掩盖了？

方法：
    1. 每个调仓日，计算因子值与「下一持有期收益」的 Spearman 相关（Rank IC）
    2. 按因子值分 10 层，等权持有到下个调仓日，看收益是否单调
    3. 按年份拆分，看 IC 的时间稳定性
    4. 多空组合（第 10 层 - 第 1 层）剥离市场 beta

收益口径与回测一致：信号日收盘出信号，次日开盘买入，下个信号日次日开盘卖出。

用法：
    python factor_ic.py --panel data_all/panel.csv.gz --universe data_all/universe.csv \
        --out reports_ic/all --min-amount 1000
    python factor_ic.py --panel data/panel.csv.gz --universe data/universe.csv \
        --out reports_ic/index --min-amount 2000
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
import factors
import run as base
from run_all import build_mask, load_data

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

GROUPS = 10
MOM_WINDOWS = [5, 10, 20, 60, 120]


def month_end(dates):
    s = pd.Series(dates, index=pd.DatetimeIndex(dates))
    return list(s.groupby([s.index.year, s.index.month]).last().sort_values())


def next_day_map(dates):
    """每个交易日 -> 下一个交易日。"""
    return {dates[i]: dates[i + 1] for i in range(len(dates) - 1)}


def build_factors(panel):
    """多窗口动量 + 低波 + 流动性，全部是「越大越好」的方向。"""
    close = factors.pivot(panel, "close").ffill()
    ret = close.pct_change(fill_method=None)

    out = {}
    for w in MOM_WINDOWS:
        out[f"动量{w}"] = close / close.shift(w) - 1
    vol = ret.rolling(60, min_periods=48).std()
    out["低波60"] = -vol          # 取负，越大越好

    # 流动性：20 日均成交额越小，流动性溢价越高，取负后越大越好
    amount = factors.pivot(panel, "amount")
    out["流动性"] = -(amount.rolling(20, min_periods=15).mean())
    return out


def analyze(panel, fac: dict, mask: pd.DataFrame):
    """对每个因子算 IC 序列与分层收益。"""
    open_ = factors.pivot(panel, "open")
    dates = list(open_.index)
    nmap = next_day_map(dates)
    rebal = [d for d in month_end(dates) if d in nmap]

    # 每个调仓日 -> 持有区间 (买入日, 卖出日)
    pairs = []
    for i in range(len(rebal) - 1):
        buy, sell = nmap[rebal[i]], nmap[rebal[i + 1]]
        pairs.append((rebal[i], buy, sell))

    result = {}
    for name, f in fac.items():
        ic_rows, group_rows = [], []
        for sig, buy, sell in pairs:
            fv = f.loc[sig]
            mv = mask.loc[sig]
            r = open_.loc[sell] / open_.loc[buy] - 1
            valid = fv.notna() & mv.fillna(False) & r.notna() & np.isfinite(r)
            if valid.sum() < 50:
                continue
            f_ok, r_ok = fv[valid], r[valid]

            # Rank IC
            ic = f_ok.rank().corr(r_ok.rank())
            ic_rows.append({"date": sig, "ic": ic})

            # 分层（1 = 因子值最小，10 = 最大）；0 = 全池等权，作为「没有因子」的对照
            try:
                g = pd.qcut(f_ok.rank(method="first"), GROUPS, labels=False) + 1
            except ValueError:
                continue
            group_rows.append({"date": sig, "group": 0, "ret": float(r_ok.mean())})
            for k in range(1, GROUPS + 1):
                idx = g[g == k].index
                group_rows.append({"date": sig, "group": k, "ret": float(r_ok[idx].mean())})

        ic_df = pd.DataFrame(ic_rows).set_index("date")
        grp_df = pd.DataFrame(group_rows)
        result[name] = {"ic": ic_df["ic"], "groups": grp_df}
    return result


def summarize(res: dict) -> pd.DataFrame:
    """IC 汇总：均值、IR、胜率、分年度均值。"""
    rows = []
    for name, r in res.items():
        ic = r["ic"].dropna()
        if len(ic) < 10:
            continue
        rows.append({
            "因子": name,
            "IC均值": ic.mean(),
            "IC标准差": ic.std(),
            "IR": ic.mean() / ic.std() if ic.std() else np.nan,
            "IC>0占比": (ic > 0).mean(),
            "样本数": len(ic),
        })
    return pd.DataFrame(rows).set_index("因子")


def yearly_ic(res: dict) -> pd.DataFrame:
    out = {}
    for name, r in res.items():
        ic = r["ic"].dropna()
        out[name] = ic.groupby(ic.index.year).mean()
    return pd.DataFrame(out)


def layered_nav(res: dict, name: str) -> pd.DataFrame:
    """把分层收益还原成净值（从 1 开始累计）。"""
    g = res[name]["groups"]
    if len(g) == 0:
        return pd.DataFrame()
    pivot = g.pivot_table(index="date", columns="group", values="ret")
    return (1 + pivot).cumprod()


def annualize(nav: pd.Series) -> float:
    """按真实日期跨度年化，不能拿月度数当交易日。"""
    if len(nav) < 2:
        return np.nan
    years = max((nav.index[-1] - nav.index[0]).days / 365.25, 0.5)
    return float(nav.iloc[-1] ** (1 / years) - 1)


def plot_layered(res, name, path):
    nav = layered_nav(res, name)
    if len(nav) == 0:
        return
    fig, ax = plt.subplots(figsize=(11, 5.5))
    cmap = plt.cm.RdYlGn
    for k in nav.columns:
        if k == 0:
            continue                      # 全池等权单独画
        ax.plot(nav.index, nav[k], label=f"第{k}层", linewidth=1.1,
                color=cmap((k - 1) / (GROUPS - 1)))
    ax.set_yscale("log")
    ax.set_title(f"{name} 分层回测（10 组等权，对数坐标）")
    ax.set_ylabel("净值（对数）")
    ax.legend(ncol=2, fontsize=8, loc="upper left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_ic(res, path):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for name in ["动量20", "动量60", "低波60", "流动性"]:
        if name in res:
            ic = res[name]["ic"].dropna()
            ax.plot(ic.index, ic.rolling(6, min_periods=3).mean(),
                    label=f"{name}（6期均值）", linewidth=1.2)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Rank IC 时间序列")
    ax.set_ylabel("IC")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main(panel_path, universe_path, out_dir, min_amount):
    os.makedirs(out_dir, exist_ok=True)
    panel, _, universe = load_data(panel_path, universe_path)
    print(f"[data] {panel['symbol'].nunique()} 只 / {len(panel)} 行")

    mask = build_mask(panel, universe, min_amount)
    fac = build_factors(panel)
    print(f"[factor] {list(fac.keys())}")

    res = analyze(panel, fac, mask)
    summary = summarize(res)
    yic = yearly_ic(res)

    summary.to_csv(os.path.join(out_dir, "ic_summary.csv"), encoding="utf-8-sig")
    yic.to_csv(os.path.join(out_dir, "ic_yearly.csv"), encoding="utf-8-sig")

    plot_layered(res, "动量20", os.path.join(out_dir, "layered_mom20.png"))
    plot_layered(res, "低波60", os.path.join(out_dir, "layered_vol60.png"))
    plot_layered(res, "流动性", os.path.join(out_dir, "layered_liq.png"))
    plot_ic(res, os.path.join(out_dir, "ic_timeseries.png"))

    # 分层年化收益表
    lines = ["# 因子有效性分析\n",
             f"- 股票池：{panel['symbol'].nunique()} 只",
             f"- 调仓：月末信号，次日开盘成交",
             f"- 分层：按因子值分 {GROUPS} 组等权\n",
             "## Rank IC 汇总\n",
             "IR = IC均值 / IC标准差。一般认为 |IR| > 0.3 且 IC>0 占比稳定高于 50% 才算可用。\n",
             base.md_table(summary, index_name="因子"),
             "\n## 分年度 IC 均值\n",
             base.md_table(yic, index_name="因子"),
             "\n## 分层年化收益\n"]

    for name in ["动量20", "低波60", "流动性"]:
        nav = layered_nav(res, name)
        if len(nav) == 0:
            continue
        ann = pd.Series({k: annualize(nav[k]) for k in nav.columns})
        tbl = ann.to_frame("年化收益")
        tbl.index = [("全池等权" if i == 0 else f"第{i}层") for i in tbl.index]
        lines.append(f"\n**{name}**\n")
        lines.append(base.md_table(tbl, index_name="分组", pct_mode="all"))

    lines.append("\n## 图表\n")
    lines.append("![动量分层](layered_mom20.png)\n")
    lines.append("![低波分层](layered_vol60.png)\n")
    lines.append("![IC](ic_timeseries.png)\n")

    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[report] {out_dir}/")
    print(summary.round(3).to_string())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--panel", default="data_all/panel.csv.gz")
    p.add_argument("--universe", default="data_all/universe.csv")
    p.add_argument("--out", default="reports_ic/all")
    p.add_argument("--min-amount", type=float, default=1000)
    a = p.parse_args()
    main(a.panel, a.universe, a.out, a.min_amount)
