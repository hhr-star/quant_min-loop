"""参数敏感性：检查策略是不是靠某组参数「调」出来的。

对因子组合 × 持仓数量做网格扫描，输出样本内/样本外对比表。
一个稳健的策略应该在参数邻域内表现连续，而不是某一组参数独好。

用法：
    python sensitivity.py
"""
import os

import pandas as pd

import backtest
import config
import factors
import metrics
from run import build_mask, load_data, md_table

REPORT_DIR = config.REPORT_DIR

FACTOR_COMBOS = {
    "三因子等权": {"momentum": 1, "low_vol": 1, "low_turnover": 1},
    "动量+低波": {"momentum": 1, "low_vol": 1, "low_turnover": 0},
    "仅动量": {"momentum": 1, "low_vol": 0, "low_turnover": 0},
    "仅低波": {"momentum": 0, "low_vol": 1, "low_turnover": 0},
}
TOP_N_GRID = [20, 50, 100]
SPLIT = pd.Timestamp(config.IN_SAMPLE_END)


def prepare(panel):
    """一次性算好三个因子的 z 分数，之后只做加权组合。"""
    fac = factors.compute_factors(panel)
    return (
        factors.cross_section_zscore(fac["momentum"]),
        factors.cross_section_zscore(-fac["volatility"]),
        factors.cross_section_zscore(-fac["avg_turnover"]),
    )


def combine(z_mom, z_vol, z_turn, w):
    score = (
        z_mom.fillna(0) * w["momentum"]
        + z_vol.fillna(0) * w["low_vol"]
        + z_turn.fillna(0) * w["low_turnover"]
    )
    valid = z_mom.notna() | z_vol.notna() | z_turn.notna()
    return score.where(valid)


def evaluate(panel, score, mask):
    res = backtest.run_backtest(panel, score, mask)
    nav = res["nav"]
    full = metrics.summarize(nav)
    oos = nav[nav.index > SPLIT]
    is_ = nav[nav.index <= SPLIT]
    return {
        "年化(全期)": metrics.annualized_return(nav),
        "夏普(全期)": metrics.sharpe(nav),
        "最大回撤": metrics.max_drawdown(nav),
        "年化(样本内)": metrics.annualized_return(is_),
        "年化(样本外)": metrics.annualized_return(oos),
        "夏普(样本外)": metrics.sharpe(oos),
        "换手": res["turnover"].mean() * metrics.TRADING_DAYS / 2,
    }


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    panel, _, universe = load_data()
    mask = build_mask(panel, universe)
    z_mom, z_vol, z_turn = prepare(panel)

    rows = []
    for name, w in FACTOR_COMBOS.items():
        score = combine(z_mom, z_vol, z_turn, w)
        for n in TOP_N_GRID:
            config.TOP_N = n
            m = evaluate(panel, score, mask)
            m.update({"因子组合": name, "持仓数": n})
            rows.append(m)
            print(f"  {name:<8} TOP{n:<4} 年化 {m['年化(全期)']:.2%} "
                  f"夏普 {m['夏普(全期)']:.2f} 样本外 {m['年化(样本外)']:.2%}", flush=True)

    # 恢复默认，避免影响其它脚本
    config.TOP_N = 50

    df = pd.DataFrame(rows).set_index(["因子组合", "持仓数"])
    df.to_csv(os.path.join(REPORT_DIR, "sensitivity.csv"), encoding="utf-8-sig")

    lines = ["# 参数敏感性\n",
             "同一份数据、同样的成本假设，只改因子组合与持仓数量。\n",
             "稳健的表现应当在参数邻域内连续；若只有某一组特别亮眼，多半是过拟合。\n"]
    pct_cols = {"年化(全期)", "年化(样本内)", "年化(样本外)", "最大回撤", "换手"}
    for col, label in [("年化(全期)", "年化收益（全期）"),
                       ("夏普(全期)", "夏普（全期）"),
                       ("年化(样本内)", "年化收益（样本内）"),
                       ("年化(样本外)", "年化收益（样本外）"),
                       ("最大回撤", "最大回撤"),
                       ("换手", "年均单边换手")]:
        sub = df[[col]].copy().reset_index()
        sub["组合"] = sub["因子组合"] + " / TOP" + sub["持仓数"].astype(str)
        sub = sub.set_index("组合")[[col]]
        lines.append(f"\n## {label}\n")
        lines.append(md_table(sub, index_name="因子组合 / 持仓数",
                              pct_mode="all" if col in pct_cols else "none"))

    path = os.path.join(REPORT_DIR, "sensitivity.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[sensitivity] {path}")


if __name__ == "__main__":
    main()
