"""第 4 周：流动性因子的落地检验。

第 2 周发现流动性因子的分层非常漂亮（第10层年化 22.72%，完美单调），
第 3 周学到「分层收益不等于组合收益」。所以这里必须回答：

    流动性溢价是真的，但它能不能在扣除真实摩擦后还剩下来？

流动性因子的本质是「买成交额小的股票」。成交额小 = 冲击成本高。
固定 10bp 滑点对这个因子是严重低估，所以这里做滑点敏感性：
滑点从 0 一路加到 100bp，看收益在哪一步消失。

用法：
    python liquidity_lab.py
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
from run_all import build_mask, load_data

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT_DIR = "reports_liq"


def build_score_liq(panel):
    """单因子：20 日均成交额越小，得分越高（流动性溢价）。"""
    amount = factors.pivot(panel, "amount")
    avg = amount.rolling(20, min_periods=15).mean()
    return factors.cross_section_zscore(-avg)


def run_variant(panel, score, mask, top_n, slippage):
    saved = (config.TOP_N, config.SLIPPAGE)
    try:
        config.TOP_N = top_n
        config.SLIPPAGE = slippage
        res = backtest.run_backtest(panel, score, mask)
        nav = res["nav"]
        return {
            "年化收益": metrics.annualized_return(nav),
            "夏普": metrics.sharpe(nav),
            "最大回撤": metrics.max_drawdown(nav),
            "年均单边换手": res["turnover"].mean() * metrics.TRADING_DAYS / 2,
            "累计成本": res["costs"].sum(),
        }
    finally:
        config.TOP_N, config.SLIPPAGE = saved


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    panel, _, universe = load_data("data_all/panel.csv.gz", "data_all/universe.csv")
    print(f"[data] {panel['symbol'].nunique()} 只")

    score = build_score_liq(panel)
    mask = build_mask(panel, universe, 1000)

    rows = []
    slips = [0.0, 0.001, 0.003, 0.005, 0.010]
    for n in [50, 100, 200]:
        for s in slips:
            r = run_variant(panel, score, mask, top_n=n, slippage=s)
            r.update({"持仓数": f"TOP{n}", "滑点": f"{s*1e4:.0f}bp"})
            rows.append(r)
            print(f"  TOP{n:<4} 滑点{s*1e4:>4.0f}bp  年化 {r['年化收益']:.2%}  "
                  f"夏普 {r['夏普']:.2f}  成本 {r['累计成本']:.1%}", flush=True)

    df = pd.DataFrame(rows).set_index(["持仓数", "滑点"])
    df.to_csv(os.path.join(OUT_DIR, "liq.csv"), encoding="utf-8-sig")

    # 图：滑点 vs 年化
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for n in ["TOP50", "TOP100", "TOP200"]:
        sub = df.loc[n]
        x = [int(i.replace("bp", "")) for i in sub.index]
        ax.plot(x, sub["年化收益"] * 100, "o-", label=n, linewidth=1.6)
    ax.axhline(11.18, color="gray", linestyle="--", linewidth=1.2,
               label="全池等权 11.18%")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("单边滑点（bp）")
    ax.set_ylabel("年化收益（%）")
    ax.set_title("流动性因子：滑点对年化收益的影响")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "slippage.png"), dpi=140)
    plt.close(fig)

    lines = ["# 第 4 周：流动性因子的落地检验\n",
             "因子：20 日均成交额越小，得分越高。\n",
             "## 全表\n",
             base.md_table(df[["年化收益", "夏普", "最大回撤", "年均单边换手", "累计成本"]],
                           index_name="持仓数 / 滑点"),
             "\n## 图\n", "![滑点](slippage.png)\n",
             "\n## 怎么读\n",
             "- 随着滑点从 0 加到 100bp，年化收益会一路下滑。",
             "- 穿过「全池等权 11.18%」那条线之后，这个因子的超额收益就不存在了。",
             "- 流动性因子选的是成交额最小的股票，真实滑点远高于 10bp。"]
    with open(os.path.join(OUT_DIR, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[report] {OUT_DIR}/")


if __name__ == "__main__":
    main()
