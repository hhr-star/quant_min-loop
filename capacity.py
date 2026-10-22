"""容量分析：多少资金还能赚到超额收益。

流动性因子赚的是「流动性折价」，代价是冲击成本。冲击成本随单笔金额
占日成交额的比例上升，业界常用的近似是平方根定律：

    滑点(bp) ≈ 250 × sqrt(p),   p = 单笔成交金额 / 日均成交额

250 这个系数是从前面的滑点敏感性反推的：
p = 1%（买 1% 的日成交额）对应约 25bp，与常见经验一致。

本脚本：
    1. 统计历史上被选中的股票，其 20 日均成交额分布
    2. 对不同资金规模，算出组合平均滑点
    3. 代入回测的「滑点-年化」曲线，得到资金-收益曲线

用法：
    python capacity.py
"""
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

OUT_DIR = "reports_capacity"
IMPACT_COEF = 250.0        # 滑点(bp) = 250 * sqrt(单笔占成交额比例)
TOP_N = 50


def build_score_liq(panel):
    amount = factors.pivot(panel, "amount")
    avg = amount.rolling(20, min_periods=15).mean()
    return factors.cross_section_zscore(-avg), avg


def selected_liquidity(panel, score, avg_amount, mask):
    """每个调仓日选出的 TOP_N 股票，记录其 20 日均成交额（万元）。"""
    dates = list(avg_amount.index)
    s = pd.Series(dates, index=pd.DatetimeIndex(dates))
    rebal = list(s.groupby([s.index.year, s.index.month]).last().sort_values())

    rows = []
    for d in rebal:
        if d not in score.index:
            continue
        valid = mask.loc[d].fillna(False)
        sc = score.loc[d][valid].dropna()
        if len(sc) == 0:
            continue
        top = sc.nlargest(min(TOP_N, len(sc)))
        amt = avg_amount.loc[d].reindex(top.index).dropna()
        amt = amt[amt > 0]
        if len(amt) == 0:
            continue
        rows.append({"date": d, "n": len(amt),
                     "amt_median": amt.median(), "amt_mean": amt.mean(),
                     "amt_q25": amt.quantile(0.25), "amt_q75": amt.quantile(0.75)})
    return pd.DataFrame(rows)


def capacity_table(sel: pd.DataFrame, nav_curve: pd.DataFrame) -> pd.DataFrame:
    """给定资金规模，算平均滑点，并从回测曲线插值出年化。"""
    slips = nav_curve["滑点bp"].values
    anns = nav_curve["年化"].values
    out = []
    for w in [1e5, 3e5, 1e6, 3e6, 1e7, 3e7, 1e8, 3e8]:
        per_stock = w / TOP_N / 1e4          # 单只金额，万元
        p = per_stock / sel["amt_median"]     # 占日成交额比例
        slip_bp = IMPACT_COEF * np.sqrt(p.clip(lower=0))
        avg_slip = float(slip_bp.median())     # 用中位数，避免极端值主导
        # 线性插值回测曲线
        ann = float(np.interp(avg_slip, slips, anns))
        out.append({"资金": w, "单只金额万": per_stock,
                    "占成交额%": float((p.median()) * 100),
                    "平均滑点bp": avg_slip, "年化": ann})
    return pd.DataFrame(out)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    panel, _, universe = load_data("data_all/panel.csv.gz", "data_all/universe.csv")
    print(f"[data] {panel['symbol'].nunique()} 只")

    score, avg_amount = build_score_liq(panel)
    mask = build_mask(panel, universe, 1000)

    sel = selected_liquidity(panel, score, avg_amount, mask)
    sel.to_csv(os.path.join(OUT_DIR, "selected_amount.csv"), index=False, encoding="utf-8-sig")
    print(f"[sel] {len(sel)} 个调仓日")
    for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
        print(f"  持仓股票 20 日均成交额 {q:.0%} 分位：{sel['amt_median'].quantile(q):.0f} 万元")

    # 回测的滑点-年化曲线（复用 liquidity_lab 的结论，直接重算 TOP50）
    curve = []
    for s in [0.0, 0.001, 0.003, 0.005, 0.010]:
        saved = (config.TOP_N, config.SLIPPAGE)
        try:
            config.TOP_N = TOP_N
            config.SLIPPAGE = s
            res = backtest.run_backtest(panel, score, mask)
            ann = metrics.annualized_return(res["nav"])
        finally:
            config.TOP_N, config.SLIPPAGE = saved
        curve.append({"滑点bp": s * 1e4, "年化": ann})
        print(f"  滑点 {s*1e4:>4.0f}bp -> 年化 {ann:.2%}", flush=True)

    nav_curve = pd.DataFrame(curve)
    cap = capacity_table(sel, nav_curve)
    cap.to_csv(os.path.join(OUT_DIR, "capacity.csv"), index=False, encoding="utf-8-sig")

    # 图
    fig, ax = plt.subplots(figsize=(9, 5))
    x = cap["资金"]
    ax.plot(x, cap["年化"] * 100, "o-", color="#c0392b", linewidth=1.8)
    ax.axhline(11.18, color="gray", linestyle="--", linewidth=1.2, label="全池等权 11.18%")
    ax.set_xscale("log")
    ax.set_xlabel("资金规模（元，对数轴）")
    ax.set_ylabel("年化收益（%）")
    ax.set_title(f"容量曲线：资金规模 vs 年化收益（流动性因子 TOP{TOP_N}）")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    for _, r in cap.iterrows():
        ax.annotate(f"{r['年化']*100:.1f}%", (r["资金"], r["年化"] * 100),
                    textcoords="offset points", xytext=(0, 9), ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "capacity.png"), dpi=140)
    plt.close(fig)

    lines = ["# 容量分析：多少资金还能赚\n",
             f"流动性因子，持仓 TOP{TOP_N}，冲击成本用平方根模型："
             f"滑点(bp) = {IMPACT_COEF:.0f} × sqrt(单只金额 / 20日均成交额)。\n",
             "## 持仓股票的成交额\n",
             base.md_table(sel[["amt_median", "amt_q25", "amt_q75"]].describe().loc[
                 ["mean", "50%", "25%", "75%"]], index_name="统计"),
             "\n## 容量表\n",
             base.md_table(cap.set_index("资金")[
                 ["单只金额万", "占成交额%", "平均滑点bp", "年化"]],
                 index_name="资金（元）"),
             "\n## 图\n", "![容量](capacity.png)\n",
             "\n## 结论\n",
             "- 资金越大，单只金额占成交额比例越高，滑点越大。",
             "- 年化跌破 11.18%（全池等权）之后，这个因子就不值得做了。",
             "- 注意：平方根模型是近似，真实冲击成本还受市场状态、下单方式影响。"]
    with open(os.path.join(OUT_DIR, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(cap.to_string(index=False))
    print(f"[report] {OUT_DIR}/")


if __name__ == "__main__":
    main()
