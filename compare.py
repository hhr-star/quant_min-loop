"""对比两个股票池的回测结果：沪深300 池 vs 全 A 池。

两者使用完全相同的因子、成本、调仓规则，唯一变量是股票池，
因此差异可以直接解释为「成分股选择偏差」的影响。

用法：
    python compare.py
"""
import os

import numpy as np
import pandas as pd

import run as base

IDX_SUMMARY = os.path.join("reports_index_2f", "summary.csv")
ALL_SUMMARY = os.path.join("reports_all", "summary.csv")
OUT_DIR = "reports_all"

METRICS = ["累计收益", "年化收益", "年化波动", "夏普比率", "最大回撤", "Calmar", "年均单边换手"]
PCT = {"累计收益", "年化收益", "年化波动", "最大回撤", "年均单边换手"}


def main():
    idx = pd.read_csv(IDX_SUMMARY, index_col=0)
    allm = pd.read_csv(ALL_SUMMARY, index_col=0)

    rows = []
    for seg in ["样本内", "样本外", "全期"]:
        for m in METRICS:
            if m not in idx.index or m not in allm.index:
                continue
            a = idx.loc[m].get(seg, np.nan)
            b = allm.loc[m].get(seg, np.nan)
            rows.append({"分段": seg, "指标": m, "沪深300池": a, "全A池": b, "差异": b - a})

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "compare.csv"), index=False, encoding="utf-8-sig")

    lines = ["# 股票池对比：沪深300 vs 全 A\n",
             "同一套因子（动量 + 低波）、同样的成本、同样的调仓规则，只换股票池。\n",
             "如果全 A 版本明显更差，说明原策略有相当一部分收益来自「成分股本身选得好」，",
             "也就是幸存者偏差，而不是因子本身有效。\n"]

    for seg in ["样本内", "样本外", "全期"]:
        sub = df[df["分段"] == seg].set_index("指标")[["沪深300池", "全A池", "差异"]]
        if len(sub) == 0:
            continue
        # 分指标决定是否按百分比展示
        raw = sub.reset_index()
        body = ["| 指标 | 沪深300池 | 全A池 | 差异 |", "|---|---|---|---|"]
        for _, r in raw.iterrows():
            if r["指标"] in PCT:
                body.append(f"| {r['指标']} | {r['沪深300池']:.2%} | {r['全A池']:.2%} | "
                            f"{r['差异']:+.2%} |")
            else:
                body.append(f"| {r['指标']} | {r['沪深300池']:.3f} | {r['全A池']:.3f} | "
                            f"{r['差异']:+.3f} |")
        lines.append(f"\n## {seg}\n")
        lines.append("\n".join(body))

    lines.append("\n## 怎么读这张表\n")
    lines.append("- **年化收益差**：负得越多，说明原策略越依赖成分股本身。")
    lines.append("- **最大回撤差**：全 A 通常回撤更大，因为小盘股波动更高。")
    lines.append("- **换手率差**：全 A 池子大，选出的票更容易被替换，换手往往更高。")

    path = os.path.join(OUT_DIR, "compare.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[compare] {path}")


if __name__ == "__main__":
    main()
