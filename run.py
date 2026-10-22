"""主流程：数据 -> 因子 -> 含成本回测 -> 样本内外评估 -> 图表与报告。

用法：
    python run.py
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

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

DATA_DIR = config.DATA_DIR
REPORT_DIR = config.REPORT_DIR


def load_data():
    panel = pd.read_csv(os.path.join(DATA_DIR, "panel.csv.gz"),
                        parse_dates=["date"], dtype={"symbol": str})
    index = pd.read_csv(os.path.join(DATA_DIR, "index.csv"), parse_dates=["date"])
    universe = pd.read_csv(os.path.join(DATA_DIR, "universe.csv"), dtype={"symbol": str})
    return panel, index, universe


def build_mask(panel: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    """可交易掩码：有行情 + 流动性达标 + 上市满 N 日 + 非 ST。

    上市天数按每只股票自己的首个交易日算：数据起点就存在的股票直接放行，
    中途出现的才当新股。否则区间开头的 N 天会把所有股票误判成新股。
    """
    amount = factors.pivot(panel, "amount")
    close = factors.pivot(panel, "close")
    avg_amount = amount.rolling(20, min_periods=10).mean()

    mask = (avg_amount >= config.MIN_AMOUNT) & close.notna()

    # 剔除前复权后出现非正价格的股票：高分红股在早期会被前复权算成负价，
    # 这是数学上正确但经济上无意义的数据，会制造出几百个百分点的假收益。
    bad = [c for c in close.columns if (close[c] <= 0).any()]
    if bad:
        mask[bad] = False
        print(f"[mask] 剔除负价格股票 {len(bad)} 只：{bad[:6]}")

    arr = close.notna().values
    n = arr.shape[0]
    first_idx = arr.argmax(axis=0)
    first_idx[~arr.any(axis=0)] = 0
    late = first_idx > 0
    allow = np.ones_like(arr, dtype=bool)
    if late.any():
        allow[:, late] = np.arange(n)[:, None] >= (first_idx[late] + config.MIN_LISTED_DAYS)[None, :]
    mask &= pd.DataFrame(allow, index=mask.index, columns=mask.columns)

    if config.EXCLUDE_ST:
        st = set(universe.loc[universe["name"].astype(str).str.contains("ST", na=False), "symbol"])
        hit = [c for c in mask.columns if c in st]
        if hit:
            mask[hit] = False
            print(f"[mask] 剔除 ST {len(hit)} 只")
    return mask


def run():
    os.makedirs(REPORT_DIR, exist_ok=True)
    panel, index_df, universe = load_data()
    print(f"[data] {panel['symbol'].nunique()} 只 / {len(panel)} 行")

    fac = factors.compute_factors(panel)
    score = factors.composite_score(fac)
    mask = build_mask(panel, universe)

    res = backtest.run_backtest(panel, score, mask)
    nav = res["nav"]
    print(f"[backtest] 净值 {nav.iloc[-1]:.3f}，调仓 {len(res['holdings'])} 次")

    # 基准：沪深300
    bench = index_df.set_index("date")["close"].reindex(nav.index).ffill()
    bench_nav = bench / bench.dropna().iloc[0]

    # 样本内外切分
    split = pd.Timestamp(config.IN_SAMPLE_END)
    segs = {
        "样本内": nav[nav.index <= split],
        "样本外": nav[nav.index > split],
        "全期": nav,
    }
    rows = {}
    for name, s in segs.items():
        if len(s) < 2:
            continue
        st = metrics.summarize(s)
        tov = res["turnover"][res["turnover"].index.isin(s.index)]
        st["年均单边换手"] = tov.mean() * metrics.TRADING_DAYS / 2
        rows[name] = st

    # 基准同期指标
    bidx = bench_nav.dropna()
    rows_bench = {}
    for name, s in segs.items():
        if len(s) < 2:
            continue
        b = bidx[bidx.index.isin(s.index)]
        if len(b) > 1:
            rows_bench[name] = metrics.summarize(b)

    summary = pd.DataFrame(rows)
    summary_bench = pd.DataFrame(rows_bench)
    summary.to_csv(os.path.join(REPORT_DIR, "summary.csv"), encoding="utf-8-sig")
    summary_bench.to_csv(os.path.join(REPORT_DIR, "summary_benchmark.csv"), encoding="utf-8-sig")

    # 分年度
    yr = metrics.yearly_returns(nav)
    yr_bench = metrics.yearly_returns(bidx)
    yearly = pd.DataFrame({"策略": yr, "沪深300": yr_bench}).dropna(how="all")
    yearly.index.name = "年份"
    yearly.to_csv(os.path.join(REPORT_DIR, "yearly.csv"), encoding="utf-8-sig")

    # ---- 图表 ----
    plot_nav(nav, bench_nav)
    plot_monthly(nav)
    plot_yearly(yearly)

    write_report(summary, summary_bench, yearly, res, panel, nav)
    print(f"[report] 已写入 {REPORT_DIR}/")


def plot_nav(nav: pd.Series, bench_nav: pd.Series):
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 1]})
    ax = axes[0]
    ax.plot(nav.index, nav, label="策略净值", linewidth=1.6, color="#c0392b")
    ax.plot(bench_nav.index, bench_nav, label="沪深300", linewidth=1.2,
            color="#7f8c8d", alpha=0.9)
    split = pd.Timestamp(config.IN_SAMPLE_END)
    ax.axvline(split, color="#2980b9", linestyle="--", linewidth=1, alpha=0.8)
    ax.text(split, ax.get_ylim()[1], "  样本外 →", color="#2980b9",
            va="top", ha="left", fontsize=10)
    ax.set_title("策略净值 vs 沪深300（含交易成本）")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.25)

    ax2 = axes[1]
    peak = nav.cummax()
    dd = nav / peak - 1
    ax2.fill_between(dd.index, dd * 100, 0, color="#c0392b", alpha=0.35)
    ax2.set_title("策略回撤（%）")
    ax2.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, "nav.png"), dpi=140)
    plt.close(fig)


def plot_monthly(nav: pd.Series):
    m = nav.resample("ME").last().pct_change(fill_method=None).dropna()
    df = pd.DataFrame({"y": m.index.year, "m": m.index.month, "r": m.values * 100})
    tab = df.pivot_table(index="y", columns="m", values="r")
    fig, ax = plt.subplots(figsize=(10, 4.2))
    data = tab.values
    vmax = np.nanmax(np.abs(data)) if np.isfinite(data).any() else 1
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(tab.shape[1]))
    ax.set_xticklabels([f"{c}月" for c in tab.columns])
    ax.set_yticks(range(tab.shape[0]))
    ax.set_yticklabels(tab.index)
    for i in range(tab.shape[0]):
        for j in range(tab.shape[1]):
            v = data[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7)
    ax.set_title("月度收益率热力图（%）")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, "monthly.png"), dpi=140)
    plt.close(fig)


def plot_yearly(yearly: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(yearly.index))
    w = 0.38
    ax.bar(x - w / 2, yearly["策略"] * 100, w, label="策略", color="#c0392b")
    ax.bar(x + w / 2, yearly["沪深300"] * 100, w, label="沪深300", color="#7f8c8d")
    ax.set_xticks(x)
    ax.set_xticklabels(yearly.index)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("分年度收益（%）")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, "yearly.png"), dpi=140)
    plt.close(fig)


PCT_ROWS = {"累计收益", "年化收益", "年化波动", "最大回撤", "日胜率", "月胜率", "年均单边换手"}


def md_table(df: pd.DataFrame, index_name: str = "指标", pct_mode: str = "auto") -> str:
    """不依赖 tabulate 的简易 Markdown 表格。

    pct_mode: auto=按指标名判断是否百分比；all=整表百分比；none=不转。
    """
    cols = [str(c) for c in df.columns]
    head = "| " + index_name + " | " + " | ".join(cols) + " |"
    sep = "|" + "---|" * (len(cols) + 1)
    rows = []
    for idx, row in df.iterrows():
        is_pct = pct_mode == "all" or (pct_mode == "auto" and str(idx) in PCT_ROWS)
        cells = []
        for v in row:
            if isinstance(v, (float, np.floating)):
                if not np.isfinite(v):
                    cells.append("-")
                elif is_pct:
                    cells.append(f"{v * 100:.2f}%")
                else:
                    cells.append(f"{v:.4f}")
            else:
                cells.append(str(v))
        rows.append("| " + str(idx) + " | " + " | ".join(cells) + " |")
    return "\n".join([head, sep] + rows)


def write_report(summary, summary_bench, yearly, res, panel, nav):
    lines = []
    lines.append("# 最小量化闭环回测报告\n")
    lines.append("## 策略设定\n")
    lines.append(f"- 股票池：沪深300 成分股（{panel['symbol'].nunique()} 只），剔除 ST 与低流动性")
    lines.append(f"- 因子：{config.MOM_WINDOW} 日动量 + {config.VOL_WINDOW} 日低波动 + "
                 f"{config.TURN_WINDOW} 日低换手，等权合成 z-score")
    lines.append(f"- 持仓：每月末打分取前 {config.TOP_N} 名，等权")
    lines.append("- 执行：月末收盘出信号，**次一交易日开盘**成交（T+1，无未来函数）")
    lines.append(f"- 成本：佣金 {config.COMMISSION*1e4:.1f}bp、印花税 {config.STAMP_TAX*1e4:.1f}bp（卖出）、"
                 f"滑点 {config.SLIPPAGE*1e4:.1f}bp\n")

    lines.append("## 绩效摘要\n")
    lines.append("**策略**\n")
    lines.append(md_table(summary))
    lines.append("\n**沪深300 基准（同期）**\n")
    lines.append(md_table(summary_bench))
    lines.append("\n## 分年度收益\n")
    lines.append(md_table(yearly, index_name="年份", pct_mode="all"))
    lines.append("\n## 交易统计\n")
    h = res["holdings"]
    if len(h):
        lines.append(f"- 调仓次数：{len(h)}")
        lines.append(f"- 平均持仓：{h['n_hold'].mean():.1f} 只")
        lines.append(f"- 累计成本占比：{res['costs'].sum():.2%}")
        lines.append(f"- 单次调仓平均成本：{h['cost'].mean():.4%}")
    lines.append("\n## 图表\n")
    lines.append("![净值](nav.png)\n")
    lines.append("![月度](monthly.png)\n")
    lines.append("![年度](yearly.png)\n")

    path = os.path.join(REPORT_DIR, "report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[report] {path}")


if __name__ == "__main__":
    run()
