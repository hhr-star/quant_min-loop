"""日频回测引擎（事件循环版，向量化取值）。

时序约束（防未来函数）：
  月末收盘 -> 用当日及之前数据算因子 -> **次一交易日开盘**成交 -> 持有到下个调仓日。

成本模型：
  买入 = 佣金 + 滑点
  卖出 = 佣金 + 印花税 + 滑点
"""
import numpy as np
import pandas as pd

import config


def _pivot(df: pd.DataFrame, col: str) -> pd.DataFrame:
    return df.pivot_table(index="date", columns="symbol", values=col, aggfunc="last")


def month_end_dates(dates: list[pd.Timestamp]) -> list[pd.Timestamp]:
    """每个月最后一个交易日。"""
    s = pd.Series(dates, index=pd.DatetimeIndex(dates))
    return list(s.groupby([s.index.year, s.index.month]).last().sort_values())


def week_end_dates(dates: list[pd.Timestamp]) -> list[pd.Timestamp]:
    """每周最后一个交易日。"""
    s = pd.Series(dates, index=pd.DatetimeIndex(dates))
    return list(s.groupby([s.index.isocalendar().year, s.index.isocalendar().week]).last().sort_values())


def _target_weights(score_row: pd.Series, tradable: pd.Series) -> pd.Series:
    """从单日打分中选出目标持仓并等权。"""
    s = score_row[tradable.reindex(score_row.index).fillna(False)].dropna()
    if len(s) == 0:
        return pd.Series(dtype=float)
    top = s.nlargest(min(config.TOP_N, len(s)))
    if config.WEIGHTING == "score":
        # 按打分加权，把负分截断到 0，避免做空
        w = top.clip(lower=0)
        if w.sum() == 0:
            w = pd.Series(1.0, index=top.index)
        return w / w.sum()
    return pd.Series(1.0 / len(top), index=top.index)


def build_signals(score: pd.DataFrame, mask: pd.DataFrame, dates: list[pd.Timestamp]) -> dict:
    """返回 {信号日: 目标权重}。信号日收盘后计算，次日开盘执行。"""
    if config.REBALANCE == "W":
        signal_days = week_end_dates(dates)
    elif config.REBALANCE == "Q":
        signal_days = month_end_dates(dates)[::3]      # 每季度调仓
    else:
        signal_days = month_end_dates(dates)

    valid_days = set(score.index)
    signals = {}
    for d in signal_days:
        if d not in valid_days:
            continue
        w = _target_weights(score.loc[d], mask.loc[d])
        if len(w):
            signals[d] = w
    return signals


def run_backtest(df: pd.DataFrame, score: pd.DataFrame, mask: pd.DataFrame) -> dict:
    """执行回测。

    返回 dict：nav(日净值)、holdings(调仓记录)、turnover(日换手)、costs(日成本)。
    """
    open_ = _pivot(df, "open")
    close = _pivot(df, "close")
    dates = list(open_.index)
    signals = build_signals(score, mask, dates)

    buy_rate = config.COMMISSION + config.SLIPPAGE
    sell_rate = config.COMMISSION + config.STAMP_TAX + config.SLIPPAGE

    nav = 1.0
    weights = pd.Series(dtype=float)       # 当前持仓权重
    navs, turnover_rec, cost_rec, hold_rec = [], [], [], []
    pending = None                          # 待执行的信号（前一交易日收盘产生）

    for i, today in enumerate(dates):
        prev = dates[i - 1] if i > 0 else None
        day_cost = 0.0
        day_turnover = 0.0

        # --- 执行调仓：上一交易日是信号日，今天开盘换仓 ---
        if prev is not None and prev in signals:
            target = signals[prev]
            # 剔除今日无法成交（停牌/一字涨停近似：开盘价缺失视为不可买）
            today_open = open_.loc[today]
            target = target[target.index.isin(today_open.dropna().index)]
            if len(target) == 0:
                target = weights.copy()     # 全部买不进就维持原仓
            else:
                target = target / target.sum()

            all_syms = weights.index.union(target.index)
            old = weights.reindex(all_syms).fillna(0.0)
            new = target.reindex(all_syms).fillna(0.0)
            buy_turn = float((new - old).clip(lower=0).sum())
            sell_turn = float((old - new).clip(lower=0).sum())
            day_turnover = (buy_turn + sell_turn) / 2
            day_cost = buy_turn * buy_rate + sell_turn * sell_rate

            nav *= (1 - day_cost)
            weights = target
            hold_rec.append({"date": today, "n_hold": len(target), "cost": day_cost, "turnover": day_turnover})

            # 换仓当日按 开盘买入 -> 收盘 计算收益
            px = close.loc[today] / today_open
            px = px.reindex(weights.index).fillna(1.0)
            port_ret = float((weights * (px - 1)).sum())
        elif i == 0 or len(weights) == 0:
            port_ret = 0.0
        else:
            # --- 普通持有日：收盘对收盘，权重随价格漂移 ---
            px = close.loc[today] / close.loc[prev]
            px = px.reindex(weights.index)
            valid = px.notna()
            if valid.sum() == 0:
                port_ret = 0.0
            else:
                w = weights[valid]
                r = px[valid] - 1
                port_ret = float((w * r).sum())
                # 漂移后的权重
                new_vals = w * (1 + r)
                tot = new_vals.sum() + (1 - weights.sum())   # 现金部分不增值
                if tot > 0:
                    weights = new_vals / tot

        nav *= (1 + port_ret)
        navs.append({"date": today, "nav": nav, "ret": port_ret, "cost": day_cost, "turnover": day_turnover})

    nav_df = pd.DataFrame(navs).set_index("date")
    return {
        "nav": nav_df["nav"],
        "returns": nav_df["ret"],
        "turnover": nav_df["turnover"],
        "costs": nav_df["cost"],
        "holdings": pd.DataFrame(hold_rec),
    }
