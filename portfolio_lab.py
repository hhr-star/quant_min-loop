"""第 3 周：组合构建实验。

回答：分层回测里「动量最低那层年化 15%」，为什么进了真实组合只剩不到 3%？

把链路拆成可控的实验，每次只动一个变量：
    1. 持仓数：500 只（接近整层） -> 10 只，看集中度的代价
    2. 成本：零成本 vs 正常成本，看摩擦吃掉多少
    3. 调仓频率：月 vs 季，看换手能不能降下来

用法：
    python portfolio_lab.py
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import backtest
import config
import factors
import metrics
import run as base
from run_all import build_mask, build_score_2f, load_data

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT_DIR = "reports_lab"


def run_variant(panel, score, mask, top_n=None, freq=None, costs=True):
    """在给定配置下跑一次回测，跑完恢复全局配置。"""
    saved = (config.TOP_N, config.REBALANCE, config.COMMISSION,
             config.STAMP_TAX, config.SLIPPAGE)
    try:
        if top_n is not None:
            config.TOP_N = top_n
        if freq is not None:
            config.REBALANCE = freq
        if not costs:
            config.COMMISSION = 0.0
            config.STAMP_TAX = 0.0
            config.SLIPPAGE = 0.0

        res = backtest.run_backtest(panel, score, mask)
        nav = res["nav"]
        tov = res["turnover"].mean() * metrics.TRADING_DAYS / 2
        return {
            "年化收益": metrics.annualized_return(nav),
            "夏普": metrics.sharpe(nav),
            "最大回撤": metrics.max_drawdown(nav),
            "年均单边换手": tov,
            "累计成本": res["costs"].sum(),
            "调仓次数": len(res["holdings"]),
        }
    finally:
        (config.TOP_N, config.REBALANCE, config.COMMISSION,
         config.STAMP_TAX, config.SLIPPAGE) = saved


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    panel, _, universe = load_data("data_all/panel.csv.gz", "data_all/universe.csv")
    print(f"[data] {panel['symbol'].nunique()} 只 / {len(panel)} 行")

    fac = factors.compute_factors(panel)
    score = build_score_2f(fac, reverse=True)      # 反转 + 低波，第 2 周验证过的方向
    mask = build_mask(panel, universe, 1000)

    rows = []

    # --- 实验 1：持仓数 ---
    print("[exp1] 持仓数 10 -> 500")
    for n in [10, 20, 50, 100, 200, 500]:
        r = run_variant(panel, score, mask, top_n=n)
        r.update({"实验": "持仓数", "设置": f"TOP{n}"})
        rows.append(r)
        print(f"  TOP{n:<4} 年化 {r['年化收益']:.2%} 夏普 {r['夏普']:.2f} "
              f"换手 {r['年均单边换手']:.0%} 成本 {r['累计成本']:.1%}", flush=True)

    # --- 实验 2：成本开关（TOP50） ---
    print("[exp2] 成本开关")
    for label, costs in [("零成本", False), ("正常成本", True)]:
        r = run_variant(panel, score, mask, top_n=50, costs=costs)
        r.update({"实验": "成本", "设置": label})
        rows.append(r)
        print(f"  {label:<6} 年化 {r['年化收益']:.2%}", flush=True)

    # --- 实验 3：调仓频率 ---
    print("[exp3] 月 / 季")
    for freq, label in [("M", "月度"), ("Q", "季度")]:
        r = run_variant(panel, score, mask, top_n=50, freq=freq)
        r.update({"实验": "频率", "设置": label})
        rows.append(r)
        print(f"  {label:<4} 年化 {r['年化收益']:.2%} 换手 {r['年均单边换手']:.0%}", flush=True)

    df = pd.DataFrame(rows).set_index(["实验", "设置"])
    df.to_csv(os.path.join(OUT_DIR, "lab.csv"), encoding="utf-8-sig")

    # --- 图：持仓数 vs 年化 ---
    e1 = df.loc["持仓数"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = [int(i.replace("TOP", "")) for i in e1.index]
    ax.plot(x, e1["年化收益"] * 100, "o-", color="#c0392b", linewidth=1.8)
    ax.set_xscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v}" for v in x])
    ax.set_xlabel("持仓数（对数轴）")
    ax.set_ylabel("年化收益（%）")
    ax.set_title("持仓数对年化收益的影响（反转 + 低波，含成本）")
    ax.grid(alpha=0.3)
    for xi, yi in zip(x, e1["年化收益"] * 100):
        ax.annotate(f"{yi:.1f}%", (xi, yi), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "topn.png"), dpi=140)
    plt.close(fig)

    lines = ["# 第 3 周：组合构建实验\n",
             "因子方向用第 2 周验证过的「反转 + 低波」。\n",
             "## 实验 1：持仓数\n",
             base.md_table(e1[["年化收益", "夏普", "最大回撤", "年均单边换手", "累计成本"]],
                           index_name="设置"),
             "\n## 实验 2：成本开关（TOP50）\n",
             base.md_table(df.loc["成本"][["年化收益", "夏普", "年均单边换手", "累计成本"]],
                           index_name="设置"),
             "\n## 实验 3：调仓频率（TOP50）\n",
             base.md_table(df.loc["频率"][["年化收益", "夏普", "年均单边换手", "累计成本"]],
                           index_name="设置"),
             "\n## 图\n", "![持仓数](topn.png)\n"]
    with open(os.path.join(OUT_DIR, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[report] {OUT_DIR}/")


if __name__ == "__main__":
    main()
